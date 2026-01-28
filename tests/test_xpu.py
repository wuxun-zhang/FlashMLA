import torch

def test_import():
    import flash_mla.xpu as flash_mla_xpu
    print("...flash_mla.xpu imported successfully")
    return

def test_sparse_prefill_fwd():
    from flash_mla import flash_mla_sparse_fwd
    device = "xpu"
    dtype = torch.bfloat16

    s_q = 1
    s_kv = 8
    h_q = 128
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 64

    q = torch.randn((s_q, h_q, d_qk), device=device, dtype=dtype)
    kv = torch.randn((s_kv, h_kv, d_qk), device=device, dtype=dtype)
    indices = torch.full((s_q, h_kv, topk), s_kv, dtype=torch.int32, device=device)

    out, max_logits, lse = flash_mla_sparse_fwd(
        q, kv, indices, sm_scale=1.0, d_v=d_v, attn_sink=None, topk_length=None
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)
    print("...sparse_prefill_fwd ran successfully")

if __name__ == "__main__":
    test_import()
    test_sparse_prefill_fwd()