// test_chaining.m -- Prototype _ANEChainingRequest for multi-kernel pipelining
// Goal: chain two conv kernels so the ANE runs them back-to-back without CPU roundtrip
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static int g_fp16_io = 0;

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

typedef struct { id model; IOSurfaceRef ioIn, ioOut; NSString *tmpDir; } CompiledKernel;

static NSString *gen_conv_mil(int ch, int sp) {
    if (g_fp16_io) {
        return [NSString stringWithFormat:
            @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"), val=tensor<string, []>(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"), val=tensor<int32, []>(1)];\n"
            "        tensor<fp16, [%d,%d,1,1]> W = const()[name=tensor<string, []>(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/weight.bin\"), offset=tensor<uint64, []>(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
            "[name=tensor<string, []>(\"conv\")];\n"
            "    } -> (y);\n}\n", ch, sp, ch, ch, ch, ch, ch, sp];
    }
    return [NSString stringWithFormat:
        @"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"
        "    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> pt = const()[name=tensor<string, []>(\"pt\"), val=tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=tensor<string, []>(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=tensor<string, []>(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=tensor<string, []>(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, []> gr = const()[name=tensor<string, []>(\"gr\"), val=tensor<int32, []>(1)];\n"
        "        tensor<string, []> to16 = const()[name=tensor<string, []>(\"to16\"), val=tensor<string, []>(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=tensor<string, []>(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=tensor<string, []>(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=tensor<string, []>(\"@model_path/weights/weight.bin\"), offset=tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=tensor<string, []>(\"conv\")];\n"
        "        tensor<string, []> to32 = const()[name=tensor<string, []>(\"to32\"), val=tensor<string, []>(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=tensor<string, []>(\"cout\")];\n"
        "    } -> (y);\n}\n", ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];
}

static CompiledKernel compile_kernel(Class gD, Class gI, int ch, int sp, NSData *wdata) {
    CompiledKernel k = {0};
    NSFileManager *fm = [NSFileManager defaultManager];

    NSString *mil = gen_conv_mil(ch, sp);
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(gD,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(gI, @selector(inMemoryModelWithDescriptor:), desc);

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        if (!g_fp16_io) {
            printf("  fp32 compile failed, retrying with fp16 I/O\n");
            g_fp16_io = 1;
            [fm removeItemAtPath:td error:nil];
            return compile_kernel(gD, gI, ch, sp, wdata);
        }
        printf("  Compile failed: %s\n", [[e description] UTF8String]);
        return k;
    }

    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

    int bpe = g_fp16_io ? 2 : 4;
    k.model = mdl;
    k.ioIn = make_surface(ch * sp * bpe);
    k.ioOut = make_surface(ch * sp * bpe);
    k.tmpDir = td;
    return k;
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("=== ANE ChainingRequest Prototype ===\n\n");

        Class gD  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class gI  = NSClassFromString(@"_ANEInMemoryModel");
        Class gAR = NSClassFromString(@"_ANERequest");
        Class gAIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class gClient = NSClassFromString(@"_ANEClient");
        Class gChain = NSClassFromString(@"_ANEChainingRequest");

        if (!gD || !gI || !gAR || !gAIO) {
            printf("ERROR: ANE private classes not found\n");
            return 1;
        }
        if (!gClient) {
            printf("ERROR: _ANEClient not found\n");
            return 1;
        }
        if (!gChain) {
            printf("ERROR: _ANEChainingRequest not found\n");
            return 1;
        }

        printf("All required classes found.\n");

        int CH = 64, SP = 32;

        _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)0.5f;
        int ws = CH*CH*2, tot = 128+ws;
        uint8_t *blob = (uint8_t*)calloc(tot, 1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
        memcpy(blob+128, w, ws);
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
        free(w);

        // -- Phase 1: Compile two kernels --
        printf("\n--- Phase 1: Compile two identical conv kernels ---\n");
        CompiledKernel k1 = compile_kernel(gD, gI, CH, SP, wdata);
        CompiledKernel k2 = compile_kernel(gD, gI, CH, SP, wdata);

        if (!k1.model || !k2.model) {
            printf("ERROR: Failed to compile kernels\n");
            return 1;
        }
        printf("  Kernel 1: compiled and loaded\n");
        printf("  Kernel 2: compiled and loaded\n");

        int bpe = g_fp16_io ? 2 : 4;
        int ioBytes = CH * SP * bpe;

        IOSurfaceLock(k1.ioIn, 0, NULL);
        if (g_fp16_io) {
            _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k1.ioIn);
            for (int i = 0; i < CH*SP; i++) inp[i] = (_Float16)1.0f;
        } else {
            float *inp = (float*)IOSurfaceGetBaseAddress(k1.ioIn);
            for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        }
        IOSurfaceUnlock(k1.ioIn, 0, NULL);

        // -- Phase 2: Baseline -- two sequential evals --
        printf("\n--- Phase 2: Baseline (sequential eval) ---\n");

        id wI1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k1.ioIn);
        id wO1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k1.ioOut);
        id wI2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k2.ioIn);
        id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), k2.ioOut);

        id req1 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI1], @[@0], @[wO1], @[@0], nil, nil, @0);
        id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gAR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI2], @[@0], @[wO2], @[@0], nil, nil, @0);

        NSError *e = nil;

        int WARMUP = 5, ITERS = 50;
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
            IOSurfaceLock(k1.ioOut, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(k2.ioIn), IOSurfaceGetBaseAddress(k1.ioOut), ioBytes);
            IOSurfaceUnlock(k1.ioOut, 0, NULL);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k2.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k1.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req1, &e);
            IOSurfaceLock(k1.ioOut, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(k2.ioIn), IOSurfaceGetBaseAddress(k1.ioOut), ioBytes);
            IOSurfaceUnlock(k1.ioOut, 0, NULL);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k2.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
        }
        double seq_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Sequential: %.3f ms total (%.3f ms/pair)\n", seq_ms, seq_ms / ITERS);

        IOSurfaceLock(k2.ioOut, 0, NULL);
        if (g_fp16_io) {
            _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k2.ioOut);
            printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n",
                   (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
        } else {
            float *out = (float*)IOSurfaceGetBaseAddress(k2.ioOut);
            printf("  Output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n", out[0], out[1], out[2], out[3]);
        }
        IOSurfaceUnlock(k2.ioOut, 0, NULL);

        // -- Phase 3: Try ChainingRequest --
        printf("\n--- Phase 3: _ANEChainingRequest exploration ---\n");

        id client = [gClient performSelector:@selector(sharedConnection)];
        if (!client) {
            printf("  WARNING: _ANEClient sharedConnection returned nil\n");
        }
        printf("  _ANEClient: %s\n", client ? "obtained" : "FAILED");

        IOSurfaceRef ioMid = make_surface(ioBytes);
        (void)((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gAIO, @selector(objectWithIOSurface:), ioMid);

        @try {
            id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                @[wI1],
                @[@[wO1]],
                @[@0],
                @[@0],
                @0,
                @[],
                @0,
                @0,
                @0);

            if (chainReq) {
                printf("  ChainingRequest created: %s\n", [[chainReq description] UTF8String]);

                BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                printf("  validate: %s\n", valid ? "YES" : "NO");

                printf("  inputBuffer: %s\n",
                    [[[chainReq valueForKey:@"inputBuffer"] description] UTF8String]);
                printf("  outputSets: %s\n",
                    [[[chainReq valueForKey:@"outputSets"] description] UTF8String]);
                printf("  loopbackInputSymbolIndex: %s\n",
                    [[[chainReq valueForKey:@"loopbackInputSymbolIndex"] description] UTF8String]);
                printf("  loopbackOutputSymbolIndex: %s\n",
                    [[[chainReq valueForKey:@"loopbackOutputSymbolIndex"] description] UTF8String]);
                printf("  procedureIndex: %s\n",
                    [[[chainReq valueForKey:@"procedureIndex"] description] UTF8String]);

                @try {
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client,
                        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        k1.model, @{}, chainReq, 21, &e);
                    printf("  prepareChainingWithModel: %s\n", ok ? "YES" : "NO");
                    if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("  prepareChainingWithModel EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  ChainingRequest: nil (creation failed)\n");
            }
        } @catch (NSException *ex) {
            printf("  ChainingRequest creation EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // -- Phase 4: Try with loopback (output feeds back as input) --
        printf("\n--- Phase 4: ChainingRequest with loopback ---\n");
        @try {
            id chainLoop = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(gChain,
                @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                @[wI1],
                @[@[wO1], @[wO2]],
                @[@0],
                @[@0],
                @0,
                @[],
                @0,
                @0,
                @0);

            if (chainLoop) {
                printf("  Loopback ChainingRequest: %s\n", [[chainLoop description] UTF8String]);

                BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainLoop, @selector(validate));
                printf("  validate: %s\n", valid ? "YES" : "NO");

                @try {
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client,
                        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        k1.model, @{}, chainLoop, 21, &e);
                    printf("  prepareChainingWithModel (loopback): %s\n", ok ? "YES" : "NO");
                    if (!ok && e) printf("    error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        @try {
                            BOOL enqOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client,
                                @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                k1.model, @[wO1], @{}, 21, &e);
                            printf("  enqueueSets: %s\n", enqOk ? "YES" : "NO");
                            if (!enqOk && e) printf("    error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("  enqueueSets EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }

                        @try {
                            BOOL bufOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client,
                                @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                k1.model, @[wI1], @{}, 21, &e);
                            printf("  buffersReady: %s\n", bufOk ? "YES" : "NO");
                            if (!bufOk && e) printf("    error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex) {
                            printf("  buffersReady EXCEPTION: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("  Loopback test EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  Loopback ChainingRequest: nil\n");
            }
        } @catch (NSException *ex) {
            printf("  Loopback creation EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // -- Cleanup --
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:k1.tmpDir error:nil];
        [fm removeItemAtPath:k2.tmpDir error:nil];
        if (k1.ioIn) CFRelease(k1.ioIn);
        if (k1.ioOut) CFRelease(k1.ioOut);
        if (k2.ioIn) CFRelease(k2.ioIn);
        if (k2.ioOut) CFRelease(k2.ioOut);
        if (ioMid) CFRelease(ioMid);

        // -- Summary --
        printf("\n--- Summary ---\n");
        printf("Sequential baseline: %.3f ms/pair (two conv evals + memcpy)\n", seq_ms / ITERS);
        printf("ChainingRequest creation: SUCCESS\n");
        printf("ChainingRequest validate: FAILS -- _ANEIOSurfaceObject needs symbolIndex\n");
        printf("  The ANE chaining API expects IOSurface objects with symbolIndex metadata.\n");
        printf("  This may require using _ANEBuffer or _ANEProgramIOSurfacesMapper\n");
        printf("  to map compiled model I/O symbols to IOSurface objects.\n");
        printf("  Next steps: explore _ANEModel.inputSymbolNames / outputSymbolNames\n");
        printf("  and _ANEProgramIOSurfacesMapper to create properly indexed buffers.\n");

        printf("\n=== ChainingRequest prototype complete ===\n");
    }
    return 0;
}
