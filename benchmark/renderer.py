"""General-purpose SVG function plotter for the LLM benchmark."""
import math

WIDTH = 1200
HEIGHT = 800

# Color palette
BG = (248, 249, 251)
GRID_MINOR = (235, 237, 241)
GRID_MAJOR = (215, 219, 225)
AXIS = (140, 148, 160)
AXIS_LABEL = (90, 98, 108)
CURVE = (24, 106, 169)


def rgb(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def world_to_pixel(x, y, x_min, x_max, y_min, y_max):
    px = (x - x_min) / (x_max - x_min) * (WIDTH - 1)
    py = (y_max - y) / (y_max - y_min) * (HEIGHT - 1)
    return px, py


def nice_tick_interval(range_val):
    """Choose a nice tick interval for the given range."""
    if range_val <= 0:
        return 1.0
    rough = range_val / 8
    mag = 10 ** math.floor(math.log10(rough))
    residual = rough / mag
    if residual <= 1.5:
        return mag
    elif residual <= 3.5:
        return 2 * mag
    elif residual <= 7.5:
        return 5 * mag
    else:
        return 10 * mag


def format_tick(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2g}"


def build_function_path(func, x_min, x_max, y_min, y_max,
                        discontinuities=None, steps=5000):
    """Build SVG path data for an arbitrary function."""
    d = []
    pen_down = False
    dx = (x_max - x_min) / steps
    eps = dx * 3
    disc = discontinuities or []
    prev_y = None
    y_range = y_max - y_min
    overflow = y_range * 0.5

    for i in range(steps + 1):
        x = x_min + i * dx

        # Skip near explicit discontinuities
        if any(abs(x - dv) < eps for dv in disc):
            pen_down = False
            prev_y = None
            continue

        try:
            y = func(x)
        except (ValueError, ZeroDivisionError, OverflowError, ArithmeticError):
            pen_down = False
            prev_y = None
            continue

        if not math.isfinite(y):
            pen_down = False
            prev_y = None
            continue

        # Detect large jumps (likely discontinuity)
        if prev_y is not None and abs(y - prev_y) > y_range * 1.5:
            pen_down = False

        # Clip: allow slight overflow for smooth curves at edges
        if y < y_min - overflow or y > y_max + overflow:
            pen_down = False
            prev_y = y
            continue

        px, py = world_to_pixel(x, y, x_min, x_max, y_min, y_max)
        cmd = "M" if not pen_down else "L"
        d.append(f"{cmd} {px:.2f} {py:.2f}")
        pen_down = True
        prev_y = y

    return " ".join(d)


def render_svg(func, x_min, x_max, y_min, y_max,
               discontinuities=None, steps=5000):
    """Render a complete SVG graph of a function.

    Returns the SVG as a string. The graph shows axes, grid lines, tick
    labels, and the function curve — but no formula or title, so an LLM
    must identify the function purely from its visual shape.
    """
    x_interval = nice_tick_interval(x_max - x_min)
    y_interval = nice_tick_interval(y_max - y_min)

    elements = []

    # Background
    elements.append(
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" '
        f'fill="{rgb(BG)}" />'
    )

    # --- Grid lines ---
    # Vertical
    x_tick = math.ceil(x_min / x_interval) * x_interval
    while x_tick <= x_max + 1e-9:
        px, _ = world_to_pixel(x_tick, 0, x_min, x_max, y_min, y_max)
        elements.append(
            f'<line x1="{px:.1f}" y1="0" x2="{px:.1f}" y2="{HEIGHT}" '
            f'stroke="{rgb(GRID_MAJOR)}" stroke-width="1" />'
        )
        x_tick += x_interval

    # Horizontal
    y_tick = math.ceil(y_min / y_interval) * y_interval
    while y_tick <= y_max + 1e-9:
        _, py = world_to_pixel(0, y_tick, x_min, x_max, y_min, y_max)
        elements.append(
            f'<line x1="0" y1="{py:.1f}" x2="{WIDTH}" y2="{py:.1f}" '
            f'stroke="{rgb(GRID_MAJOR)}" stroke-width="1" />'
        )
        y_tick += y_interval

    # --- Axes ---
    # Y-axis (x=0)
    if x_min <= 0 <= x_max:
        x0, _ = world_to_pixel(0, 0, x_min, x_max, y_min, y_max)
        elements.append(
            f'<line x1="{x0:.1f}" y1="0" x2="{x0:.1f}" y2="{HEIGHT}" '
            f'stroke="{rgb(AXIS)}" stroke-width="2.5" />'
        )
    # X-axis (y=0)
    if y_min <= 0 <= y_max:
        _, y0 = world_to_pixel(0, 0, x_min, x_max, y_min, y_max)
        elements.append(
            f'<line x1="0" y1="{y0:.1f}" x2="{WIDTH}" y2="{y0:.1f}" '
            f'stroke="{rgb(AXIS)}" stroke-width="2.5" />'
        )

    # --- Function curve ---
    path_data = build_function_path(
        func, x_min, x_max, y_min, y_max, discontinuities, steps
    )
    if path_data:
        elements.append(
            f'<path d="{path_data}" fill="none" stroke="{rgb(CURVE)}" '
            f'stroke-width="3.5" stroke-linecap="round" '
            f'stroke-linejoin="round" />'
        )

    # --- Tick labels ---
    # X-axis labels
    x_tick = math.ceil(x_min / x_interval) * x_interval
    # Find where to place the labels vertically
    if y_min <= 0 <= y_max:
        _, label_base_y = world_to_pixel(0, 0, x_min, x_max, y_min, y_max)
        label_y = min(label_base_y + 22, HEIGHT - 8)
    else:
        label_y = HEIGHT - 8

    while x_tick <= x_max + 1e-9:
        if abs(x_tick) > 1e-9:  # skip origin label
            px, _ = world_to_pixel(x_tick, 0, x_min, x_max, y_min, y_max)
            label = format_tick(x_tick)
            elements.append(
                f'<text x="{px:.1f}" y="{label_y:.1f}" '
                f'text-anchor="middle" fill="{rgb(AXIS_LABEL)}" '
                f'font-size="15" font-family="Helvetica, Arial, sans-serif">'
                f'{label}</text>'
            )
        x_tick += x_interval

    # Y-axis labels
    y_tick = math.ceil(y_min / y_interval) * y_interval
    if x_min <= 0 <= x_max:
        label_base_x, _ = world_to_pixel(0, 0, x_min, x_max, y_min, y_max)
        label_x = max(label_base_x + 8, 8)
    else:
        label_x = 8

    while y_tick <= y_max + 1e-9:
        if abs(y_tick) > 1e-9:  # skip origin
            _, py = world_to_pixel(0, y_tick, x_min, x_max, y_min, y_max)
            label = format_tick(y_tick)
            elements.append(
                f'<text x="{label_x:.1f}" y="{py + 5:.1f}" '
                f'fill="{rgb(AXIS_LABEL)}" font-size="15" '
                f'font-family="Helvetica, Arial, sans-serif">'
                f'{label}</text>'
            )
        y_tick += y_interval

    # Origin label "0"
    if x_min <= 0 <= x_max and y_min <= 0 <= y_max:
        x0, y0 = world_to_pixel(0, 0, x_min, x_max, y_min, y_max)
        elements.append(
            f'<text x="{x0 + 8:.1f}" y="{y0 + 20:.1f}" '
            f'fill="{rgb(AXIS_LABEL)}" font-size="15" '
            f'font-family="Helvetica, Arial, sans-serif">0</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">\n'
        + "\n".join(f"  {e}" for e in elements)
        + "\n</svg>\n"
    )
    return svg
