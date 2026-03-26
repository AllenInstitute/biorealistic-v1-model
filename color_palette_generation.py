# %% Generate color palette

import colorsys, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
import pandas as pd

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def rel_lum(rgb):
    """WCAG relative luminance for an sRGB triple (0-1 floats)."""
    linear = lambda c: c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
    r, g, b = map(linear, rgb)
    return 0.2126*r + 0.7152*g + 0.0722*b


def best_sv_for_lum(hue_deg: float,
                    base_s   : float,
                    target_L : float,
                    v_low    : float = 0.00,
                    eps      : float = 1e-3):
    """
    Search a grid of S,V values around `base_s` and return the tuple
    (s, v, luminance) that:
       1) Minimises |L − target_L|
       2) Among ties (|ΔL| < eps) minimises |s − base_s|
    Parameters
    ----------
    hue_deg : float     – hue in degrees 0-360
    target_L: float     – desired luminance (0-1)
    base_s  : float     – preferred saturation starting point
    s_span  : float     – search ±s_span around base_s  (default ±0.40)
    v_low   : float     – lowest V value to consider     (default 0.60)
    eps     : float     – luminance tolerance regarded as “equally good”
    """

    best       = None          # (ΔL, |ΔS|, s, v, L)
    L_best     = 1.0           # initialise high
    sat_best   = 1.0
    
    s = base_s

    for v in np.linspace(v_low, 1.0, 49):
        L = rel_lum(colorsys.hsv_to_rgb(hue_deg/360, s, v))
        dL = abs(L - target_L)
        if dL < L_best - eps:          # strictly better luminance
            L_best, sat_best = dL, abs(s - base_s)
            best = (s, v, L)
        elif abs(dL - L_best) < eps:   # equal luminance → prefer sat close to base
            dS = abs(s - base_s)
            if dS < sat_best:
                sat_best, best = dS, (s, v, L)

    return best  # (s, v, L)
# def best_sv_for_lum(hue_deg, base_s, target_l):
#     """Find (s, v) near base_s giving luminance closest to target_l."""
#     best_s, best_v, best_diff = base_s, 1.0, 1
#     for s in np.linspace(max(0, base_s-0.4), min(1, base_s+0.4), 1001):
#         for v in np.linspace(0.60, 1.00, 41):
#             L = rel_lum(colorsys.hsv_to_rgb(hue_deg/360, s, v))
#             diff = abs(L - target_l)
#             if diff < best_diff:
#                 best_s, best_v, best_diff = s, v, diff
#             if diff < 1e-3:        # good enough
#                 return best_s, best_v, L
#     return best_s, best_v, rel_lum(colorsys.hsv_to_rgb(hue_deg/360, best_s, best_v))

# ------------------------------------------------------------
# Palette builder
# ------------------------------------------------------------
def build_palette(target_L=0.55, sats=None):
    """
    Returns a palette dict mapping (layer, subtype) -> hex, and a
    table list if you want to inspect HSV/L values.
    """
    layers = ["L2/3", "L4", "L5", "L6"]
    # high-saturation in L2/3 → low in L6
    if sats is None:
        base_sat = {"L2/3": 0.70, "L4": 0.60, "L5": 0.50, "L6": 0.40}
    else:
        base_sat = sats
    main_hues = {"Exc": 0, "PV": 90, "SST": 180, "VIP": 270}

    palette, table = {}, []

    # Add L1 Inh entry (gray with target luminance)
    # For gray, we use saturation = 0, and find the value that gives target luminance
    gray_v = target_L ** (1/2.4)  # Approximate inverse gamma correction for gray
    if gray_v > 1.0:
        gray_v = 1.0
    gray_rgb = (gray_v, gray_v, gray_v)
    gray_L = rel_lum(gray_rgb)
    gray_hex = '#{:02X}{:02X}{:02X}'.format(*(int(c*255) for c in gray_rgb))
    palette[("L1", "Inh")] = gray_hex
    table.append(("L1", "Inh", None, 0.0, round(gray_v*100,1), round(gray_L,3), gray_hex))

    for ly in layers:
        s0 = base_sat[ly]
        for ct, h in main_hues.items():
            s, v, L = best_sv_for_lum(h, s0, target_L)
            rgb = colorsys.hsv_to_rgb(h/360, s, v)
            hexcol = '#{:02X}{:02X}{:02X}'.format(*(int(c*255) for c in rgb))
            palette[(ly, ct)] = hexcol
            table.append((ly, ct, h, round(s*100,1), round(v*100,1), round(L,3), hexcol))

    # L5 Exc sub-subtypes (E5, ET, IT, NP)
    ext_hues = {"E5": 0, "ET": 30, "IT": 10, "NP": 330}
    for lab, h in ext_hues.items():
        s, v, L = best_sv_for_lum(h, base_sat["L5"], target_L)
        rgb = colorsys.hsv_to_rgb(h/360, s, v)
        hexcol = '#{:02X}{:02X}{:02X}'.format(*(int(c*255) for c in rgb))
        palette[("L5", lab)] = hexcol
        table.append(("L5", lab, h, round(s*100,1), round(v*100,1), round(L,3), hexcol))

    return palette, table

# ------------------------------------------------------------
# Plotter
# ------------------------------------------------------------
def plot_palette(pal, title="Palette"):
    layers   = ["L2/3", "L4", "L5", "L6"]
    cols     = ["Exc", "PV", "SST", "VIP"]
    cell_w, cell_h = 1.8, 1.1
    fig_w, fig_h   = cell_w*len(cols), cell_h*len(layers)+0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')

    # draw grid
    for r, ly in enumerate(layers):
        for c, ct in enumerate(cols):
            x0, y0 = c*cell_w, (len(layers)-1-r)*cell_h
            ax.add_patch(patches.Rectangle((x0,y0), cell_w, cell_h,
                                           facecolor='white', edgecolor='black'))
            if (ly, ct) in pal:
                col = pal[(ly, ct)]
                # txt = 'white' if sum(int(col[i:i+2],16) for i in (1,3,5)) < 384 else 'black'
                txt = 'white'
                ax.add_patch(patches.Rectangle((x0,y0), cell_w, cell_h,
                                               facecolor=col, edgecolor='black'))
                if ly == "L5" and ct == "Exc":
                    continue
                ax.text(x0+cell_w/2, y0+cell_h/2, ct, ha='center', va='center',
                        fontsize=9, weight='bold', color=txt)
        # layer label
        ax.text(-0.15, (len(layers)-1-r)*cell_h + cell_h/2, ly,
                ha='right', va='center', fontsize=10, weight='bold')

    # column labels
    for c, ct in enumerate(cols):
        ax.text(c*cell_w + cell_w/2, fig_h-0.2, ct, ha='center', va='bottom',
                fontsize=10, weight='bold')

    # L5 Exc split
    exc_x    = cols.index("Exc") * cell_w
    base_y   = (len(layers)-1-layers.index("L5")) * cell_h
    left_w   = cell_w * 0.5
    stripe_h = cell_h / 3
    ax.add_patch(patches.Rectangle((exc_x, base_y), left_w, cell_h,
                                   facecolor=pal[("L5","Exc")], edgecolor='black'))
    ax.text(exc_x+left_w/2, base_y+cell_h/2, "Exc", ha='center', va='center',
            fontsize=8, weight='bold', color='white')

    for i, lab in enumerate(["ET","IT","NP"][::-1]):      # NP bottom
        y  = base_y + i*stripe_h
        col = pal[("L5", lab)]
        txt = 'white'
        ax.add_patch(patches.Rectangle((exc_x+left_w, y), left_w, stripe_h,
                                       facecolor=col, edgecolor='black'))
        ax.text(exc_x+0.75*cell_w, y+stripe_h/2, lab, ha='center', va='center',
                fontsize=7, weight='bold', color=txt)

    plt.title(title, fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Usage examples
# ------------------------------------------------------------
# Dark raster palette (L≈0.18)
dark_sats = {"L2/3": 0.80, "L4": 0.70, "L5": 0.60, "L6": 0.50}
dark_pal, dark_table = build_palette(target_L=0.164, sats=dark_sats)
plot_palette(dark_pal, "Raster palette  (L ≈ 0.164)")
palette_df = pd.DataFrame(dark_table, columns=["Layer", "Subtype", "Hue", "S", "V", "L", "Hex"])
print(palette_df)
palette_df.to_csv("dark_palette.csv", index=False)

# %%

# Bright box-plot palette (L≈0.55)
bright_sats = {"L2/3": 0.25, "L4": 0.20, "L5": 0.15, "L6": 0.10}
bright_pal, bright_table = build_palette(target_L=0.60, sats=bright_sats)
plot_palette(bright_pal, "Box-plot palette (L ≈ 0.60)")
palette_df = pd.DataFrame(bright_table, columns=["Layer", "Subtype", "Hue", "S", "V", "L", "Hex"])
print(palette_df)
palette_df.to_csv("bright_palette.csv", index=False)


# %% unified palette
unified_sats = {"L2/3": 0.6, "L4": 0.5, "L5": 0.4, "L6": 0.3}
unified_pal, unified_table = build_palette(target_L=0.25, sats=unified_sats)
plot_palette(unified_pal, "Unified palette (L ≈ 0.30)")
pd.DataFrame(unified_table, columns=["Layer", "Subtype", "Hue", "S", "V", "L", "Hex"])




# %% format the table
dark_pal, dark_table = build_palette(target_L=0.18)
plot_palette(dark_pal, "Raster palette  (L ≈ 0.18)")
