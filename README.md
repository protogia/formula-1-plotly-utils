# Formula-1-plotly-utils

Utilities to plot formula1-data based on fastf1-api via plotly. Contains transfered examples from [fastf1-gallery](https://docs.fastf1.dev/gen_modules/examples_gallery/index.html) as well as own creations used in my [just-for-fun-evaluations](https://github.com/protogia/formula1-evaluations).

## Install
```bash
poetry add git+https://github.com/protogia/formula-1-plotly-utils.git@main
poetry install
poetry run python -c "import utils_lib; print(formula_1_plotly_utils.__version__)"
```

## Usage
```py
from formula_1_plotly_utils import utils as f1p
import fastf1
import fastf1.plotting

# Load data from fastf1
session = fastf1.get_session(2023, 'Zandvoort', 'Q')
session.load()

fig = utils.plot_tyre_strategies(
    drivers=session.laps['Driver'].unique(),
    laps=session.laps,
    track_status=session.track_status
)

fig.show()
```