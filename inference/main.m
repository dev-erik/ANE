// main.m -- Qwen2.5-0.5B inference on Apple Neural Engine
// Supports three modes:
//   1. Single-shot:  ./qwen_ane weights.bin "token_ids" [max_tokens]
//   2. Stdin server:  ./qwen_ane weights.bin --server
//   3. Socket server: ./qwen_ane weights.bin --server /tmp/qwen_ane.sock
//
// Build:
//   xcrun clang -O2 -framework Foundation -framework IOSurface \
//     -framework CoreML -framework Accelerate -ldl -lobjc -fobjc-arc \
//     -o qwen_ane main.m
//
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include "qwen_ane_infer.h"

int g_fp16_io = 0;
static QwenModel g_model;
static const char *g_sock_path = NULL;

static void cleanup_socket(void) {
    if (g_sock_path) unlink(g_sock_path);
}

static void handle_signal(int sig) {
    (void)sig;
    cleanup_socket();
    _exit(0);
}

static int load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    int config[7];
    fread(config, sizeof(int), 7, f);
    int dim = config[0], hidden = config[1], n_layers = config[2];
    int n_heads = config[3], n_kv_heads = config[4], vocab = config[5];
    printf("Config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d\n",
           dim, hidden, n_layers, n_heads, n_kv_heads, vocab);

    int q_dim = n_heads * QWEN_HEAD_DIM;
    int kv_dim = n_kv_heads * QWEN_HEAD_DIM;

    g_model.embed = (float*)malloc((size_t)vocab * dim * sizeof(float));
    fread(g_model.embed, sizeof(float), (size_t)vocab * dim, f);

    for (int l = 0; l < n_layers; l++) {
        g_model.rms_att[l] = (float*)malloc(dim * sizeof(float));
        fread(g_model.rms_att[l], sizeof(float), dim, f);

        g_model.wq[l] = (float*)malloc((size_t)q_dim * dim * sizeof(float));
        fread(g_model.wq[l], sizeof(float), (size_t)q_dim * dim, f);
        g_model.wk[l] = (float*)malloc((size_t)kv_dim * dim * sizeof(float));
        fread(g_model.wk[l], sizeof(float), (size_t)kv_dim * dim, f);
        g_model.wv[l] = (float*)malloc((size_t)kv_dim * dim * sizeof(float));
        fread(g_model.wv[l], sizeof(float), (size_t)kv_dim * dim, f);
        g_model.wo[l] = (float*)malloc((size_t)q_dim * dim * sizeof(float));
        fread(g_model.wo[l], sizeof(float), (size_t)dim * q_dim, f);

        g_model.q_bias[l] = (float*)malloc(q_dim * sizeof(float));
        g_model.k_bias[l] = (float*)malloc(kv_dim * sizeof(float));
        g_model.v_bias[l] = (float*)malloc(kv_dim * sizeof(float));
        fread(g_model.q_bias[l], sizeof(float), q_dim, f);
        fread(g_model.k_bias[l], sizeof(float), kv_dim, f);
        fread(g_model.v_bias[l], sizeof(float), kv_dim, f);

        g_model.rms_ffn[l] = (float*)malloc(dim * sizeof(float));
        fread(g_model.rms_ffn[l], sizeof(float), dim, f);

        g_model.w_gate[l] = (float*)malloc((size_t)hidden * dim * sizeof(float));
        fread(g_model.w_gate[l], sizeof(float), (size_t)hidden * dim, f);
        g_model.w_up[l] = (float*)malloc((size_t)hidden * dim * sizeof(float));
        fread(g_model.w_up[l], sizeof(float), (size_t)hidden * dim, f);
        g_model.w_down[l] = (float*)malloc((size_t)dim * hidden * sizeof(float));
        fread(g_model.w_down[l], sizeof(float), (size_t)dim * hidden, f);
    }

    g_model.rms_final = (float*)malloc(dim * sizeof(float));
    fread(g_model.rms_final, sizeof(float), dim, f);

    long file_size = ftell(f);
    fclose(f);
    printf("Weights loaded (%.0f MB)\n", (float)file_size / 1024 / 1024);
    return 0;
}

// Parse space-separated token IDs from a string. Returns count.
static int parse_tokens(const char *str, int *ids, int max_ids) {
    int n = 0;
    char *buf = strdup(str);
    char *saveptr;
    char *p = strtok_r(buf, " \t\n\r", &saveptr);
    while (p && n < max_ids) {
        ids[n++] = atoi(p);
        p = strtok_r(NULL, " \t\n\r", &saveptr);
    }
    free(buf);
    return n;
}

static double timespec_diff(struct timespec *a, struct timespec *b) {
    return (b->tv_sec - a->tv_sec) + (b->tv_nsec - a->tv_nsec) / 1e9;
}

// Run one generation pass. Writes output token IDs to out_ids, returns count.
// If out_fd >= 0, writes formatted results there; otherwise prints to stdout.
static int generate(int *prompt_ids, int n_prompt, int max_gen,
                    int *out_ids, int max_out,
                    double *prefill_tps, double *decode_tps) {
    struct timespec t0, t1, t_pre;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int next = 0;
    for (int i = 0; i < n_prompt; i++)
        next = qwen_forward(&g_model, prompt_ids[i]);

    clock_gettime(CLOCK_MONOTONIC, &t_pre);
    double ps = timespec_diff(&t0, &t_pre);
    *prefill_tps = ps > 0 ? n_prompt / ps : 0;

    int eos = 151645, eos2 = 151643;
    int n_out = 0;
    for (int i = 0; i < max_gen && n_out < max_out; i++) {
        if (n_out < max_out) out_ids[n_out++] = next;
        if (next == eos || next == eos2) break;
        next = qwen_forward(&g_model, next);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ds = timespec_diff(&t_pre, &t1);
    int gen_tokens = n_out > 1 ? n_out - 1 : 0;
    *decode_tps = ds > 0 ? gen_tokens / ds : 0;

    return n_out;
}

// --- Stdin server mode ---
static void run_stdin_server(void) {
    printf("READY\n");
    fflush(stdout);

    char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        // Format: "token_id token_id ... [|max_tokens]"
        int max_gen = 50;
        char *pipe = strchr(line, '|');
        if (pipe) {
            max_gen = atoi(pipe + 1);
            *pipe = '\0';
        }

        int prompt_ids[2048];
        int n_prompt = parse_tokens(line, prompt_ids, 2048);
        if (n_prompt == 0) {
            printf("ERR: empty prompt\n");
            fflush(stdout);
            continue;
        }

        int out_ids[4096];
        double p_tps, d_tps;
        int n_out = generate(prompt_ids, n_prompt, max_gen, out_ids, 4096, &p_tps, &d_tps);

        printf("OUT:");
        for (int i = 0; i < n_out; i++) printf(" %d", out_ids[i]);
        printf("\n");
        printf("PERF: prefill=%.1f decode=%.1f prompt=%d gen=%d\n",
               p_tps, d_tps, n_prompt, n_out);
        fflush(stdout);

        qwen_reset(&g_model);
    }
}

// --- Socket server mode ---
static void run_socket_server(const char *sock_path) {
    g_sock_path = sock_path;
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    atexit(cleanup_socket);

    unlink(sock_path);

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(srv); return;
    }
    if (listen(srv, 4) < 0) {
        perror("listen"); close(srv); return;
    }

    printf("Listening on %s\n", sock_path);
    printf("READY\n");
    fflush(stdout);

    while (1) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) { perror("accept"); continue; }

        // Read request: {"tokens": [1,2,3], "max_tokens": 50}
        char buf[131072];
        ssize_t total = 0;
        while (total < (ssize_t)sizeof(buf) - 1) {
            ssize_t n = read(client, buf + total, sizeof(buf) - 1 - total);
            if (n <= 0) break;
            total += n;
            if (memchr(buf, '\n', total) || memchr(buf, '}', total)) break;
        }
        buf[total] = '\0';

        // Minimal JSON parsing for {"tokens": [...], "max_tokens": N}
        int prompt_ids[2048];
        int n_prompt = 0;
        int max_gen = 50;

        char *tok_start = strstr(buf, "\"tokens\"");
        if (tok_start) {
            char *bracket = strchr(tok_start, '[');
            if (bracket) {
                char *p = bracket + 1;
                while (*p && *p != ']' && n_prompt < 2048) {
                    while (*p && (*p == ' ' || *p == ',')) p++;
                    if (*p == ']') break;
                    prompt_ids[n_prompt++] = (int)strtol(p, &p, 10);
                }
            }
        }

        char *mt = strstr(buf, "\"max_tokens\"");
        if (mt) {
            char *colon = strchr(mt, ':');
            if (colon) max_gen = (int)strtol(colon + 1, NULL, 10);
        }

        if (n_prompt == 0) {
            const char *err = "{\"error\": \"no tokens\"}\n";
            write(client, err, strlen(err));
            close(client);
            continue;
        }

        int out_ids[4096];
        double p_tps, d_tps;
        int n_out = generate(prompt_ids, n_prompt, max_gen, out_ids, 4096, &p_tps, &d_tps);

        // Build JSON response
        char resp[131072];
        int off = snprintf(resp, sizeof(resp),
            "{\"output\": [");
        for (int i = 0; i < n_out; i++)
            off += snprintf(resp + off, sizeof(resp) - off,
                "%s%d", i ? ", " : "", out_ids[i]);
        off += snprintf(resp + off, sizeof(resp) - off,
            "], \"prefill_tps\": %.1f, \"decode_tps\": %.1f, "
            "\"prompt_tokens\": %d, \"gen_tokens\": %d}\n",
            p_tps, d_tps, n_prompt, n_out);

        write(client, resp, off);
        close(client);

        printf("[socket] prompt=%d gen=%d prefill=%.1f decode=%.1f t/s\n",
               n_prompt, n_out, p_tps, d_tps);
        fflush(stdout);

        qwen_reset(&g_model);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr,
                "Usage:\n"
                "  %s <weights.bin> \"token_ids\" [max_tokens]   (single-shot)\n"
                "  %s <weights.bin> --server                     (stdin loop)\n"
                "  %s <weights.bin> --server /tmp/qwen_ane.sock  (socket server)\n",
                argv[0], argv[0], argv[0]);
            return 1;
        }

        printf("=== Qwen2.5-0.5B ANE Inference ===\n\n");

        setbuf(stdout, NULL);

        printf("Loading weights...\n");
        if (load_weights(argv[1]) != 0) return 1;

        qwen_alloc(&g_model);

        printf("Compiling ANE kernels (169 total)...\n");
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        qwen_compile_kernels(&g_model);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double compile_sec = timespec_diff(&t0, &t1);
        printf("Compile time: %.1fs\n\n", compile_sec);

        // Check for --server flag
        int server_mode = 0;
        const char *sock_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--server") == 0) {
                server_mode = 1;
                if (i + 1 < argc && argv[i+1][0] != '-')
                    sock_path = argv[++i];
            }
        }

        if (server_mode) {
            if (sock_path)
                run_socket_server(sock_path);
            else
                run_stdin_server();
            return 0;
        }

        // Single-shot mode (original behavior)
        if (argc < 3) {
            fprintf(stderr, "Error: provide token IDs or --server\n");
            return 1;
        }

        int max_gen = 50;
        if (argc >= 4 && strcmp(argv[3], "--server") != 0)
            max_gen = atoi(argv[3]);

        int prompt_ids[2048];
        int n_prompt = parse_tokens(argv[2], prompt_ids, 2048);
        printf("Prompt: %d tokens, generating up to %d\n", n_prompt, max_gen);

        int out_ids[4096];
        double p_tps, d_tps;
        int n_out = generate(prompt_ids, n_prompt, max_gen, out_ids, 4096, &p_tps, &d_tps);

        printf("OUT:");
        for (int i = 0; i < n_out; i++) printf(" %d", out_ids[i]);
        printf("\n");

        printf("\nPrefill: %.1f t/s (%d tokens)\n", p_tps, n_prompt);
        printf("Decode:  %.1f t/s (%d tokens)\n", d_tps, n_out > 1 ? n_out - 1 : 0);

        return 0;
    }
}
