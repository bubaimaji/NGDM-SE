import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange

N_FFT = 512
HOP = 160
WIN = 400
HID_FEATURE = 64
BETA = 2.0


def mag_phase_stft(y, n_fft=N_FFT, hop_size=HOP, win_size=WIN, compress_factor=0.3, center=True):
    hann = torch.hann_window(win_size).to(y.device)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann,
                       center=center, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.pow(torch.abs(spec), compress_factor)
    pha = torch.angle(spec)
    return mag, pha


def mag_phase_istft(mag, pha, n_fft=N_FFT, hop_size=HOP, win_size=WIN, compress_factor=0.3, center=True):
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann = torch.hann_window(win_size).to(com.device)
    return torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann, center=center)



def small_init_(p, dim):
    torch.nn.init.normal_(p, mean=0.0, std=math.sqrt(2/(5*dim)))

def wang_init_(p, dim, n):
    torch.nn.init.normal_(p, mean=0.0, std=2/n/math.sqrt(dim))

def bias_linspace_init_(p, start=3.0, end=6.0):
    with torch.no_grad():
        p.copy_(torch.linspace(start, end, p.shape[0]))

def parallel_stabilized_simple(q, k, v, ig, fg, ltr=None, eps=1e-6):
    B, NH, S, DH = q.shape
    log_fg = F.logsigmoid(fg)
    if ltr is None or S < ltr.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=q.device))
    cum = torch.cat([torch.zeros((B, NH, 1, 1), device=q.device), torch.cumsum(log_fg, dim=-2)], dim=-2)
    rep = cum.repeat(1, 1, 1, S+1)
    lfm = torch.where(ltr, (rep - rep.transpose(-2,-1))[:, :, 1:, 1:], -float("inf"))
    logD = lfm + ig.transpose(-2,-1)
    maxD, _ = torch.max(logD, dim=-1, keepdim=True)
    D = torch.exp(logD - maxD)
    C = (q @ (k/math.sqrt(DH)).transpose(-2,-1)) * D
    norm = torch.maximum(C.sum(dim=-1, keepdim=True).abs(), torch.exp(-maxD))
    return (C/(norm+eps)) @ v

class LinearHeadwiseExpand(nn.Module):
    def __init__(self, dim, heads, bias=False):
        super().__init__()
        self.heads = heads
        dph = dim // heads
        self.weight = nn.Parameter(torch.empty(heads, dph, dph))
        self.bias = nn.Parameter(torch.empty(dim)) if bias else None
        nn.init.normal_(self.weight.data, std=math.sqrt(2/5/dph))
        if self.bias is not None: nn.init.zeros_(self.bias.data)
    def forward(self, x):
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.heads)
        x = einops.einsum(x, self.weight, "... nh d, nh o d -> ... nh o")
        x = einops.rearrange(x, "... nh o -> ... (nh o)")
        return x + self.bias if self.bias is not None else x

class CausalConv1d(nn.Module):
    def __init__(self, dim, k=4, bias=True):
        super().__init__()
        self.pad = k-1
        self.conv = nn.Conv1d(dim, dim, k, padding=self.pad, groups=dim, bias=bias)
    def forward(self, x):
        x = einops.rearrange(x, "b l d -> b d l")
        return einops.rearrange(self.conv(x)[:,:,:-self.pad], "b d l -> b l d")

class LNNoBias(nn.Module):
    def __init__(self, ndim, bias=False, residual=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.ndim, self.residual = ndim, residual
    def forward(self, x):
        w = 1.0 + self.weight if self.residual else self.weight
        return F.layer_norm(x, (self.ndim,), w, self.bias, 1e-5)

class MHLayerNorm(LNNoBias):
    def forward(self, x):
        B, NH, S, DH = x.shape
        gn = x.transpose(1,2).reshape(B*S, NH*DH)
        w = 1.0 + self.weight if self.residual else self.weight
        out = F.group_norm(gn, NH, w, self.bias, 1e-5)
        return out.view(B,S,NH,DH).transpose(1,2)

class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.igate = nn.Linear(3*dim, heads)
        self.fgate = nn.Linear(3*dim, heads)
        self.outnorm = MHLayerNorm(dim, bias=True)
        self._cache = {}
        nn.init.zeros_(self.fgate.weight); bias_linspace_init_(self.fgate.bias)
        nn.init.zeros_(self.igate.weight); nn.init.normal_(self.igate.bias, std=0.1)
    def forward(self, q, k, v):
        B, S, _ = q.shape
        z = torch.cat([q,k,v], dim=-1)
        q = q.view(B,S,self.heads,-1).transpose(1,2)
        k = k.view(B,S,self.heads,-1).transpose(1,2)
        v = v.view(B,S,self.heads,-1).transpose(1,2)
        ig = self.igate(z).transpose(-1,-2).unsqueeze(-1)
        fg = self.fgate(z).transpose(-1,-2).unsqueeze(-1)
        key = (S, str(q.device))
        if key not in self._cache:
            self._cache[key] = torch.tril(torch.ones(S,S,dtype=torch.bool,device=q.device))
        h = parallel_stabilized_simple(q,k,v,ig,fg,self._cache[key])
        return self.outnorm(h).transpose(1,2).reshape(B,S,-1)

class MLSTMCore(nn.Module):
    def __init__(self, dim, expansion=2, qkv_block=4, conv_k=4):
        super().__init__()
        self.dim = dim
        inner = expansion*dim
        heads = inner // qkv_block
        self.proj_up = nn.Linear(dim, 2*inner)
        self.q_proj = LinearHeadwiseExpand(inner, heads)
        self.k_proj = LinearHeadwiseExpand(inner, heads)
        self.v_proj = LinearHeadwiseExpand(inner, heads)
        self.conv = CausalConv1d(inner, conv_k)
        self.cell = MatrixLSTMCell(inner, qkv_block)
        self.skip = nn.Parameter(torch.ones(inner))
        self.proj_down = nn.Linear(inner, dim)
        small_init_(self.proj_up.weight, dim); nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim, 1); nn.init.zeros_(self.proj_down.bias)
        for p in (self.q_proj,self.k_proj,self.v_proj): small_init_(p.weight, dim)
    def forward(self, x):
        xi = self.proj_up(x)
        xm, z = torch.chunk(xi, 2, dim=-1)
        xc = F.silu(self.conv(xm))
        q,k,v = self.q_proj(xc), self.k_proj(xc), self.v_proj(xm)
        h = self.cell(q,k,v)
        h = h + self.skip*xc
        h = h * F.silu(z)
        return self.proj_down(h)

class XLSTMBlock(nn.Module):
    """dim -> dim, [B,S,D], TIME-AXIS ONLY --fast mechanism."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.core = MLSTMCore(dim)
    def forward(self, x):
        return x + self.core(self.norm(x))


class LearnableSigmoid2D(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

def get_padding_2d(k, d=(1,1)):
    return (int((k[0]*d[0]-d[0])/2), int((k[1]*d[1]-d[1])/2))

class DenseBlock(nn.Module):
    def __init__(self, hid, kernel_size=(3,3), depth=2):   # depth reduced 4->2 to cut MACs
                                                                                                                  
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dil = 2**i
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hid*(i+1), hid, kernel_size, dilation=(dil,1), padding=get_padding_2d(kernel_size,(dil,1))),
                nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid)))
    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.blocks[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    """NOTE: reduces FREQUENCY only (stride=(1,2)); TIME is preserved."""
    def __init__(self, in_ch, hid):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_ch, hid, 1), nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid))
        self.db = DenseBlock(hid, depth=2)
        self.c2 = nn.Sequential(nn.Conv2d(hid, hid, (1,3), stride=(1,2)), nn.InstanceNorm2d(hid, affine=True), nn.PReLU(hid))
    def forward(self, x):
        return self.c2(self.db(self.c1(x)))

class MagDecoder(nn.Module):
    """NOTE: restores FREQUENCY only; assumes TIME dim is unchanged from encoder input."""
    def __init__(self, hid, out_ch, n_fft, beta):
        super().__init__()
        self.db = DenseBlock(hid, depth=2)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hid, hid, (1,3), stride=(1,2)),
            nn.Conv2d(hid, out_ch, 1), nn.InstanceNorm2d(out_ch, affine=True), nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 1))
        self.lsig = LearnableSigmoid2D(n_fft//2+1, beta=beta)
    def forward(self, x):
        x = self.db(x)
        x = self.conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsig(x)
        return rearrange(x, 'b f t -> b t f').unsqueeze(1)


# ============ Fusion ============

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.s_attends_n = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.n_attends_s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_n = nn.LayerNorm(dim)
        self.fuse = nn.Linear(dim*2, dim)
    def forward(self, hs, hn):
        sa, w1 = self.s_attends_n(hs, hn, hn, need_weights=False)
        na, w2 = self.n_attends_s(hn, hs, hs, need_weights=False)
        hs2 = self.norm_s(hs + sa)
        hn2 = self.norm_n(hn + na)
        return self.fuse(torch.cat([hs2, hn2], dim=-1)), None

class ConcatFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Linear(dim*2, dim)
    def forward(self, hs, hn):
        return self.fuse(torch.cat([hs, hn], dim=-1)), None


# ============ NGDM-SE: DenseEncoder/MagDecoder + fast time-only dual xLSTM ============

class NGDM_SE(nn.Module):
    def __init__(self, hid_feature=HID_FEATURE,
                 use_noise_routing=True, use_dual_memory=True, use_attention_fusion=True):
        super().__init__()
        self.use_noise_routing = use_noise_routing
        self.use_dual_memory = use_dual_memory
        self.use_attention_fusion = use_attention_fusion

        self.dense_encoder = DenseEncoder(in_ch=1, hid=hid_feature)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 200, N_FFT//2+1)   # [B,1,T,F]
            o = self.dense_encoder(dummy)
            _, C, T_probe, Freq = o.shape
            feat_dim = C * Freq
        self.C, self.Freq = C, Freq

        self.proj = nn.Sequential(nn.Linear(feat_dim, hid_feature), nn.LayerNorm(hid_feature), nn.GELU())
        self.unproj = nn.Linear(hid_feature, feat_dim)   # back to spatial for MagDecoder

        self.gate = nn.Sequential(nn.Linear(hid_feature, hid_feature), nn.Sigmoid())

        if use_dual_memory:
            self.speech_xlstm = XLSTMBlock(hid_feature)
            self.noise_xlstm = XLSTMBlock(hid_feature)
            self.fusion = CrossAttentionFusion(hid_feature) if use_attention_fusion else ConcatFusion(hid_feature)
        else:
            self.shared_xlstm = XLSTMBlock(hid_feature)

        self.mask_decoder = MagDecoder(hid_feature, out_ch=1, n_fft=N_FFT, beta=BETA)

    def forward(self, noisy, return_attention=False):
        squeeze_back = False
        if noisy.dim() == 3:
            noisy = noisy.squeeze(1)
            squeeze_back = True

        orig_len = noisy.shape[-1]
        mag, pha = mag_phase_stft(noisy)
        x = rearrange(mag, 'b f t -> b t f').unsqueeze(1)   # [B,1,T,F]

        feat = self.dense_encoder(x)   # [B,C,T,Freq_reduced] -- T UNCHANGED, only Freq reduced
        B, C, T, Freq = feat.shape

        z = feat.permute(0, 2, 1, 3).reshape(B, T, C*Freq)   # [B, T, C*Freq] -- flatten per timestep
        z = self.proj(z)   # [B, T, hid]

        if self.use_noise_routing:
            g = self.gate(z.mean(dim=1)).unsqueeze(1)
        else:
            g = 1.0

        if self.use_dual_memory:
            hs = self.speech_xlstm(z)
            hn = self.noise_xlstm(z * g)
            fused, _ = self.fusion(hs, hn)
        else:
            fused = self.shared_xlstm(z * g if self.use_noise_routing else z)

        fused = self.unproj(fused)                      # [B, T, C*Freq]
        fused = fused.view(B, T, C, Freq).permute(0, 2, 1, 3)   # back to [B,C,T,Freq]

        mask = self.mask_decoder(fused)   # [B,1,T,F_full]
        denoised_mag = rearrange(mask * x, 'b c t f -> b f t c').squeeze(-1)

        enhanced = mag_phase_istft(denoised_mag, pha)

        if enhanced.shape[-1] > orig_len:
            enhanced = enhanced[..., :orig_len]
        elif enhanced.shape[-1] < orig_len:
            enhanced = F.pad(enhanced, (0, orig_len - enhanced.shape[-1]))

        if squeeze_back:
            enhanced = enhanced.unsqueeze(1)
        return enhanced


if __name__ == "__main__":
    import time
    x = torch.randn(2, 40000)

    print("=== DenseEncoder/MagDecoder + fast time-only dual xLSTM (with TimeReduce/Restore) ===")
    model = NGDM_SE()
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n/1e6:.3f} M")

    t0 = time.time()
    y = model(x)
    t1 = time.time()
    assert y.shape == x.shape
    print(f"shapes OK: {x.shape} -> {y.shape}")
    print(f"Single CPU forward pass: {(t1-t0)*1000:.0f} ms")

    try:
        from thop import profile
        model_cpu = NGDM_SE().eval()
        dummy = torch.randn(1, 16000)
        macs, params = profile(model_cpu, inputs=(dummy,), verbose=False)
        print(f"\nMACs (1s @16kHz input): {macs/1e9:.3f} G  (target: <15-20G)")
        print(f"FLOPs (approx 2x MACs): {macs*2/1e9:.3f} G")
    except ImportError:
        print("\n[thop not installed -- run: pip install thop --break-system-packages]")
    except Exception as e:
        print(f"\n[thop profiling failed: {e} -- likely complex-STFT tracing limitation, not a model bug]")

    print("\n=== Ablation: w/o Noise Routing ===")
    m2 = NGDM_SE(use_noise_routing=False)
    print(f"Params: {sum(p.numel() for p in m2.parameters())/1e6:.3f} M")
    assert m2(x).shape == x.shape

    print("\n=== Ablation: w/o Dual Memory ===")
    m3 = NGDM_SE(use_dual_memory=False)
    print(f"Params: {sum(p.numel() for p in m3.parameters())/1e6:.3f} M")
    assert m3(x).shape == x.shape

    print("\n=== Ablation: w/o Attention Fusion ===")
    m4 = NGDM_SE(use_attention_fusion=False)
    print(f"Params: {sum(p.numel() for p in m4.parameters())/1e6:.3f} M")
    assert m4(x).shape == x.shape

    print("\nAll model variants.")
