def test_cuda():
    import torch
    cuda_version = torch.version.cuda
    cuda_available = torch.cuda.is_available()
    return f"CUDA Version: {cuda_version}, CUDA Available: {cuda_available}"

test_cuda_result = test_cuda()
print(test_cuda_result)