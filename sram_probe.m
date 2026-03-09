#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static int g_fp16_io = 0;  // M1/M2: cast op unsupported, use fp16 I/O directly

static NSData *buildWeightBlob(int ch) {
    NSUInteger wsize = (NSUInteger)ch * ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE;
    chunk[4]=0x01; chunk[10]=0x08;
    uint16_t *fp16 = (uint16_t*)(chunk + 64);
    for (NSUInteger j = 0; j < (NSUInteger)ch * ch; j++)
        fp16[j] = (arc4random() & 0x03FF) | 0x2000;
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *genMIL(int ch, int sp) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.0)\n[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n{\n"];
    if (g_fp16_io) {
        [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ch, sp];
    } else {
        [m appendFormat:@"    func main<ios16>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    }
    [m appendString:
        @"        tensor<string, []> c_pad_type = const()[name = tensor<string, []>(\"c_pad_type\"), val = tensor<string, []>(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = tensor<string, []>(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = tensor<string, []>(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = tensor<string, []>(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, []> c_groups = const()[name = tensor<string, []>(\"c_groups\"), val = tensor<int32, []>(1)];\n"];
    NSString *xIn;
    if (g_fp16_io) {
        xIn = @"x";
    } else {
        [m appendString:@"        tensor<string, []> to_fp16 = const()[name = tensor<string, []>(\"to_fp16\"), val = tensor<string, []>(\"fp16\")];\n"];
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = tensor<string, []>(\"cast_in\")];\n", ch, sp];
        xIn = @"x16";
    }
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(64)))];\n", ch, ch, ch, ch];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = %@)[name = tensor<string, []>(\"conv\")];\n", ch, sp, xIn];
    if (g_fp16_io) {
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = identity(x = y16)[name = tensor<string, []>(\"out\")];\n", ch, sp];
    } else {
        [m appendString:@"        tensor<string, []> to_fp32 = const()[name = tensor<string, []>(\"to_fp32\"), val = tensor<string, []>(\"fp32\")];\n"];
        [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = tensor<string, []>(\"cast_out\")];\n", ch, sp];
    }
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

double bench(int ch, int sp) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *milData = [[genMIL(ch, sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb = buildWeightBlob(ch);

        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) return -2;

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return -3;

        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:tmpDir error:nil];
            if (!g_fp16_io) {
                printf("[ANE] fp32 compile failed, retrying with fp16 I/O (M1/M2 fallback)\n");
                g_fp16_io = 1;
                return bench(ch, sp);
            }
            return -4;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:tmpDir error:nil]; return -5;
        }

        size_t bpe = g_fp16_io ? 2 : 4;
        NSUInteger bytes = ch * sp * bpe;
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        for (int i = 0; i < 5; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

        int iters = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double ms = ticksToMs(mach_absolute_time() - t0) / iters;

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];
        return ms;
    }
}

int main() {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    printf("=== ANE SRAM Fine Probe (weights only vary, spatial=64) ===\n\n");
    printf("%-12s %8s %10s %8s %12s\n", "Channels", "W (MB)", "ms/eval", "TFLOPS", "GFLOPS/MB");
    printf("--------------------------------------------------------------\n");

    int chs[] = {256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 8192};
    int sps[] = {64,  64,  64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   32};

    for (int i = 0; i < 13; i++) {
        int ch = chs[i], sp = sps[i];
        double w_mb = (double)ch * ch * 2 / 1024 / 1024;
        double gf = 2.0 * ch * ch * sp / 1e9;
        double ms = bench(ch, sp);
        double tf = (ms > 0) ? gf / ms : 0;
        double eff = (ms > 0) ? tf * 1000 / w_mb : 0;
        printf("%6d ch   %7.1f  %8.3f ms %7.2f  %10.1f %s\n",
               ch, w_mb, ms, tf, eff,
               (i > 0 && eff < 100) ? " <-- spilling?" : "");
    }
    return 0;
}
