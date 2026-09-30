# Formula-1-plotly-utils

Utilities to plot formula1-data based on fastf1-api via plotly. Contains transfered matplotlib-examples from [fastf1-gallery](https://docs.fastf1.dev/gen_modules/examples_gallery/index.html) as well as own creations used in my just for fun [formula1-evaluations](https://github.com/protogia/formula1-evaluations).

## Install
```bash
poetry add git+https://github.com/protogia/formula-1-plotly-utils.git@main
poetry install
poetry run python -c "import formula_1_plotly_utils; print(formula_1_plotly_utils.__version__)"
```

## Usage
```py
import fastf1
from formula_1_plotly_utils import ChartMarker, apply_driver_colors
from formula_1_plotly_utils.plot import strategy, overview

# Load data from fastf1
session = fastf1.get_session(2024, 'São Paulo', 'R')
session.load()

# apply fastf1 driver colors once, all charts use them
laps = apply_driver_colors(session.laps)            # colormap='official' for brighter colors
results = apply_driver_colors(session.results, session)

fig = strategy.plot_tyre_strategies(
    laps=laps,
    track_status=session.track_status,
    drivers=['VER', 'OCO', 'GAS', 'RUS'],
    title='Brazil 2024: Strategy in the Rain',
    markers=[
        ChartMarker('COL crash', lap=31, driver='COL'),
        {'label': 'Restart', 'time': '02:23:12'},   # dicts work as well
    ],
)
fig.show()
```

## Conventions
All plot functions share the same behaviour:

- **Layout:** one light theme based on plotly's default look (`_core/layout.py`), title + session subtitle, axis titles with units as `Label [unit]`.
- **Colors:** driver colors from fastf1 (`get_driver_style`), teammates get a dashed line. Categorical driver axes print the driver codes in black with a border in the driver color. In the telemetry comparison, drivers with similar colors (e.g. teammates) get two distinct standard colors; the team color is then shown next to the name in the legend. Compound colors come from fastf1 as well.
- **`title`:** every function accepts a custom title.
- **`markers`:** every function accepts a `ChartMarker`, a dict with the same keys or a list of both. The position used depends on the chart's x-axis:

  | Field      | Used on                                                         |
  |------------|-----------------------------------------------------------------|
  | `lap`      | lap axes (converted to time on time axes)                       |
  | `time`     | time axes; session time as timedelta, `'HH:MM:SS'` or minutes (converted to laps on lap axes) |
  | `distance` | telemetry / track charts, meters                                 |
  | `driver`   | driver axes (bar/box charts); pins the marker to the driver's line in line charts |
  | `y`        | explicit vertical position                                      |
  | `color`    | marker color (default white)                                    |

- **Track status:** charts with a `track_status` parameter mark SC, VSC and red flag phases as translucent bands with a label; the moment of *VSC ending* is a dotted line.
