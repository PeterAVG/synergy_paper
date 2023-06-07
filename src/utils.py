from typing import Any


def _set_font_size(ax: Any, misc: int = 26, legend: int = 20) -> None:
    try:
        _ = len(ax)
    except TypeError:
        ax = [ax]
    for _ax in ax:
        for item in (
            [_ax.title, _ax.xaxis.label, _ax.yaxis.label]
            + _ax.get_xticklabels()
            + _ax.get_yticklabels()
        ):
            item.set_fontsize(misc)
    for _ax in ax:
        try:
            for item in _ax.get_legend().get_texts():
                item.set_fontsize(legend)
        except AttributeError:
            pass
