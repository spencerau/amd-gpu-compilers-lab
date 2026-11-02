import triton

def bench(name, fn, bytes_moved, flops=0):
    ms = triton.testing.do_bench(fn, return_mode="median")
    line = f"{name:<24} {ms:8.3f} ms  {bytes_moved / (ms * 1e6):8.1f} GB/s"
    if flops:
        line += f"  {flops / (ms * 1e9):8.2f} TFLOP/s"
    print(line)
    return ms