/* ============================================================================
 *  empty_driverlib_main.c —— MNIST CNN (32x32) F28P55 NPU 板端程序 (golden + 交互)
 *
 *  模型: mnist_cnn (CNN_TWO 工程; Conv3x3 x3 + MaxPool x3 + FC, TI-QAT int8)
 *        输入 (1,1,32,32) float32 [0,1] (白字黑底) -> 输出 (1,10) logits (float)
 *        NPU 库: artifacts/mod.a  (卸载: FCONV/GCONV x2/FC x2/MAXPOOL x3)
 *
 *  功能:
 *   [1] 开机: 0.5s 延时 -> 冷启动计时 -> golden 自检一遍 (12 张 MNIST, 输出同旧版)
 *   [2] 之后进入交互模式: 轮询串口, 收到 PC 帧即推理并回发结果
 *       协议: PC -> 板  `0xA5 0x5A` + 1024 字节 uint8 像素 (32x32 行主序, 白字黑底;
 *                        板端 /255.0f 还原 [0,1], 与训练/手写板口径一致)
 *             板 -> PC  文本: "[PC] pred=<d> conf=<f>% t_avg=<f>us t_min=<f>us\r\n"
 *                              "     logits: <10 个 3 位小数>\r\n"
 *             (每次响应用 PC_REPS 次推理取 avg/min)
 *   [3] 耗时口径: tvmgen_default_run() + while(!tvmgen_default_finished); CPUTimer1 @150MHz
 *
 *  注意: 不要调用 TI_NPU_init() (新版 mod.a 运行库内部已初始化; 调用会链接报错)
 * ==========================================================================*/
#include "board.h"
#include "device.h"
#include "driverlib.h"
#include <stdint.h>
#include <math.h>

#include "artifacts/tvmgen_default.h"   /* NPU 模型接口 (mod.a 提供) */
#include "golden_mnist.h"               /* 12 个 golden 样本 (export_golden.py 生成) */

#define N_IN         (GOLD_IMG_W * GOLD_IMG_H)   /* 1024 */
#define N_OUT        10
#define SYSCLK_MHZ   150.0f
#define WARMUP_RUNS  2
#define REPEAT_RUNS  32
#define PC_REPS      4                            /* 交互帧: 计时推理次数 */

/* ------------------- 全局变量 (CCS Expressions 窗口可 Watch) ------------------- */
static float g_in[N_IN];                    /* 推理输入缓冲 (32x32, [0,1]) */
static float g_out[N_OUT];                  /* 推理输出缓冲 (10 logits)   */
static struct tvmgen_default_inputs  g_npu_in;
static struct tvmgen_default_outputs g_npu_out;

volatile float g_us_cold    = -1.0f;        /* 冷启动(上电后第一次)耗时 us */
volatile float g_us_min     = 1.0e9f;       /* golden 轮统计 min us */
volatile float g_us_avg     = 0.0f;         /* golden 轮统计 avg us */
volatile float g_us_max     = 0.0f;         /* golden 轮统计 max us */
volatile float g_gold_maxd  = -1.0f;        /* 板端 int8 vs PC float 的 max|d| */
volatile int   g_ok_count   = 0;            /* golden 轮 argmax 匹配数 */
static   int   g_tcnt       = 0;
static   float g_tsum       = 0.0f;

/* PC 帧接收 (交互模式) */
static uint8_t g_rxpix[N_IN];
static int  g_rxstate = 0;                  /* 0=找 0xA5, 1=等 0x5A, 2=收像素 */
static int  g_rxlen   = 0;
volatile int   g_pc_frames  = 0;            /* 已处理的 PC 帧数 */
volatile float g_us_pc_last = -1.0f;        /* 最近一次 PC 帧的 t_avg (us) */

/* ------------------- 串口小工具 (避免引入 printf 依赖) ------------------- */
static void uart_putc(char c) { SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)c); }
static void uart_puts(const char *s) { while (*s) { uart_putc(*s++); } }

static void uart_u32(uint32_t v)
{
    char b[11];
    int n = 0;
    if (!v) { uart_putc('0'); return; }
    while (v) { b[n++] = (char)('0' + v % 10u); v /= 10u; }
    while (n) { uart_putc(b[--n]); }
}

/* 定点打印小数: dec = 1..3 位 */
static void uart_fx(float v, int dec)
{
    uint32_t mul = (dec == 1) ? 10u : (dec == 2) ? 100u : 1000u;
    int32_t s;
    if (v < 0.0f) { uart_putc('-'); v = -v; }
    s = (int32_t)(v * (float)mul + 0.5f);
    uart_u32((uint32_t)(s / (int32_t)mul));
    uart_putc('.');
    s = s % (int32_t)mul;
    if (dec >= 3 && s < 100) { uart_putc('0'); }
    if (dec >= 2 && s < 10)  { uart_putc('0'); }
    uart_u32((uint32_t)s);
}

/* 读一个已到达的字节; 无数据返回 -1 (非阻塞) */
static int uart_getc(void)
{
    if (SCI_getRxFIFOStatus(mySCI0_BASE) == SCI_FIFO_RX0) { return -1; }
    return (int)SCI_readCharBlockingFIFO(mySCI0_BASE);
}

/* ------------------- 计时器 / 推理 ------------------- */
static void CPUTimer1_init(void)
{
    CPUTimer_stopTimer(CPUTIMER1_BASE);
    CPUTimer_setPeriod(CPUTIMER1_BASE, 0xFFFFFFFFu);
    CPUTimer_setPreScaler(CPUTIMER1_BASE, 0u);
    CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);
    CPUTimer_startTimer(CPUTIMER1_BASE);
}

/* NPU 推理一次; 返回耗时 (us); 口径 = run + 等待完成 */
static float npu_infer(void)
{
    uint32_t t0 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
    (void)tvmgen_default_run(&g_npu_in, &g_npu_out);
    while (!tvmgen_default_finished)
    {
        /* 轮询等待 NPU 子图完成 (完成标志由 NPU 中断服务程序置位) */
    }
    uint32_t t1 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
    return (float)(t0 - t1) / SYSCLK_MHZ;
}

static int argmax10(const float *v)
{
    int i, bi = 0;
    for (i = 1; i < N_OUT; i++) { if (v[i] > v[bi]) { bi = i; } }
    return bi;
}

static float softmax_conf10(const float *v)
{
    float e[N_OUT], sum = 0.0f, m = v[0];
    int i, bi;
    for (i = 1; i < N_OUT; i++) { if (v[i] > m) { m = v[i]; } }
    for (i = 0; i < N_OUT; i++) { e[i] = expf(v[i] - m); sum += e[i]; }
    bi = argmax10(v);
    return e[bi] / sum;
}

/* 装载 golden 样本 s 到输入缓冲: 原始 MNIST uint8 像素 / 255 -> [0,1] */
static void load_sample(int s)
{
    int i;
    for (i = 0; i < N_IN; i++) { g_in[i] = (float)GOLD_PIX[s][i] / 255.0f; }
}

/* 打印 32x32 ASCII 图 (golden 样本 s) */
static void print_art(int s)
{
    int y, x, i = 0;
    for (y = 0; y < GOLD_IMG_H; y++)
    {
        for (x = 0; x < GOLD_IMG_W; x++, i++)
        {
            uint8_t v = GOLD_PIX[s][i];
            uart_putc((v > 160) ? '#' : ((v > 48) ? '+' : '.'));
        }
        uart_puts("\r\n");
    }
}

/* ------------------- 交互模式: 处理一帧 PC 数据 ------------------- */
static void handle_pc_frame(void)
{
    int i, r, pred;
    float tsum = 0.0f, tmin = 1.0e9f, tavg, conf;

    for (i = 0; i < N_IN; i++) { g_in[i] = (float)g_rxpix[i] / 255.0f; }

    for (r = 0; r < PC_REPS; r++)
    {
        float t = npu_infer();
        tsum += t;
        if (t < tmin) { tmin = t; }
    }
    tavg = tsum / (float)PC_REPS;

    pred = argmax10(g_out);
    conf = softmax_conf10(g_out);
    g_pc_frames++;
    g_us_pc_last = tavg;

    uart_puts("[PC] pred=");
    uart_u32((uint32_t)pred);
    uart_puts(" conf=");
    uart_fx(conf * 100.0f, 1);
    uart_puts("% t_avg=");
    uart_fx(tavg, 1);
    uart_puts("us t_min=");
    uart_fx(tmin, 1);
    uart_puts("us\r\n     logits: ");
    for (i = 0; i < N_OUT; i++)
    {
        uart_fx(g_out[i], 3);
        uart_putc(' ');
    }
    uart_puts("\r\n");
}

/* 接收状态机 (主循环反复调用, 非阻塞) */
static void rx_poll(void)
{
    int c;
    while ((c = uart_getc()) >= 0)
    {
        switch (g_rxstate)
        {
            case 0:
                if (c == 0xA5) { g_rxstate = 1; }
                break;
            case 1:
                if (c == 0x5A) { g_rxstate = 2; g_rxlen = 0; }
                else if (c != 0xA5) { g_rxstate = 0; }
                break;
            default:                          /* 收像素 */
                g_rxpix[g_rxlen++] = (uint8_t)c;
                if (g_rxlen >= N_IN) { g_rxstate = 0; handle_pc_frame(); }
                break;
        }
    }
}

/* ------------------- golden 自检 (开机跑一遍) ------------------- */
static void run_golden_suite(int run_no)
{
    int s, r, i, ok, ok_expect_true;
    float tsum, tmin, tmax, tavg, gmaxd, dmax;

    uart_puts("\r\n============ RUN #");
    uart_u32((uint32_t)run_no);
    uart_puts(" ============\r\n");
    uart_puts("MNIST CNN NPU board test | model: mnist_cnn TI-QAT int8 (mod.a)\r\n");
    uart_puts("input 1x1x32x32 float [0,1] | output 1x10 logits | REPEAT=");
    uart_u32(REPEAT_RUNS);
    uart_puts(" reps/sample\r\n");
    uart_puts("timing = tvmgen_default_run + wait loop, CPUTimer1 @150MHz\r\n");
    uart_puts("cold-start first inference: ");
    uart_fx(g_us_cold, 1);
    uart_puts(" us\r\n");

    ok = 0; ok_expect_true = 0; gmaxd = 0.0f;
    g_tsum = 0.0f; g_tcnt = 0; g_us_min = 1.0e9f; g_us_max = 0.0f;

    for (s = 0; s < GOLD_N_SAMPLES; s++)
    {
        int pred;
        float conf;

        load_sample(s);

        /* 预热 2 次 (不计时), 再计时 REPEAT 次 */
        for (r = 0; r < WARMUP_RUNS; r++) { (void)npu_infer(); }
        tsum = 0.0f; tmin = 1.0e9f; tmax = 0.0f;
        for (r = 0; r < REPEAT_RUNS; r++)
        {
            float t = npu_infer();
            if (t < tmin) { tmin = t; }
            if (t > tmax) { tmax = t; }
            tsum += t;
        }
        tavg = tsum / (float)REPEAT_RUNS;
        if (tmin < g_us_min) { g_us_min = tmin; }
        if (tmax > g_us_max) { g_us_max = tmax; }
        g_tsum += tsum;
        g_tcnt += REPEAT_RUNS;

        pred = argmax10(g_out);
        conf = softmax_conf10(g_out);
        if (pred == GOLD_EXPECT[s]) { ok++; }
        if (GOLD_EXPECT[s] == GOLD_TRUE[s]) { ok_expect_true++; }

        dmax = 0.0f;
        for (i = 0; i < N_OUT; i++)
        {
            float d = fabsf(g_out[i] - GOLD_LOGITS[s][i]);
            if (d > dmax) { dmax = d; }
        }
        if (dmax > gmaxd) { gmaxd = dmax; }

        if (s == 0)
        {
            uart_puts("     input art of S00 ('#'>160, '+' >48, else '.'):\r\n");
            print_art(0);
        }

        uart_puts("[S");
        uart_u32((uint32_t)s);
        uart_puts("] true=");
        uart_u32(GOLD_TRUE[s]);
        uart_puts(" expect=");
        uart_u32(GOLD_EXPECT[s]);
        uart_puts(" pred=");
        uart_u32((uint32_t)pred);
        uart_puts((pred == GOLD_EXPECT[s]) ? " OK" : " **MISMATCH**");
        uart_puts(" | t_avg=");
        uart_fx(tavg, 1);
        uart_puts("us min=");
        uart_fx(tmin, 1);
        uart_puts(" max=");
        uart_fx(tmax, 1);
        uart_puts("\r\n     conf=");
        uart_fx(conf * 100.0f, 1);
        uart_puts("%  vsPC_maxd=");
        uart_fx(dmax, 3);
        uart_puts("  logits: ");
        for (i = 0; i < N_OUT; i++)
        {
            uart_fx(g_out[i], 3);
            uart_putc(' ');
        }
        uart_puts("\r\n");
    }

    g_us_avg = g_tsum / (float)g_tcnt;
    g_ok_count = ok;
    g_gold_maxd = gmaxd;

    uart_puts("---- SUMMARY RUN #");
    uart_u32((uint32_t)run_no);
    uart_puts(" ----\r\n");
    uart_puts("argmax match (pred==PC expect): ");
    uart_u32((uint32_t)ok);
    uart_puts("/12 | expect==true: ");
    uart_u32((uint32_t)ok_expect_true);
    uart_puts("/12\r\n");
    uart_puts("timing n=");
    uart_u32((uint32_t)g_tcnt);
    uart_puts(": min=");
    uart_fx(g_us_min, 1);
    uart_puts("us avg=");
    uart_fx(g_us_avg, 1);
    uart_puts("us max=");
    uart_fx(g_us_max, 1);
    uart_puts("us\r\n");
    uart_puts("board int8 vs PC float logits: max|d|=");
    uart_fx(g_gold_maxd, 3);
    uart_puts(" (quantization noise only)\r\n");
}

/* ------------------- 主程序 ------------------- */
void main(void)
{
    /* 1. 系统初始化 (与工程原有流程一致; device.c 已含 NPU 时钟使能) */
    Device_init();
    Interrupt_initModule();
    Interrupt_initVectorTable();
    Board_init();                       /* SysConfig 生成 (SCIA 115200-8N1) */
    EINT;
    ERTM;

    /* 2. 计时器 (上电一次) */
    CPUTimer1_init();

    /* 注意: 不要调用 TI_NPU_init()! 新版 mod.a 运行库内部已初始化 */

    /* 3. 绑定模型输入输出指针 */
    g_npu_in.input   = (void *)g_in;
    g_npu_out.output = (void *)g_out;

    /* 4. 留 0.5s 给 XDS110 虚拟串口枚举 */
    DEVICE_DELAY_US(500000);

    /* 5. 冷启动计时 (上电后第一次推理) */
    load_sample(0);
    g_us_cold = npu_infer();

    /* 6. 开机 golden 自检一遍 */
    run_golden_suite(1);

    /* 7. 交互模式: 等待 PC 帧 (A5 5A + 1024B) */
    uart_puts("\r\n==== INTERACTIVE MODE READY ====\r\n");
    uart_puts("send frame: A5 5A + 1024 bytes uint8 (32x32, ink=bright)\r\n");

    while (1)
    {
        rx_poll();
    }
}
