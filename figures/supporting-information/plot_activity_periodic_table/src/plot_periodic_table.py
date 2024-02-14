#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This module was taken from: https://github.com/arosen93/ptable_trends, and only minor adjustment was made by myself.  --Haoyu Yang.
"""


from bokeh.io import export_png
from bokeh.io import show as show_
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge
import colorcet
from matplotlib import rcParams
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import coolwarm, ScalarMappable, turbo
from pandas import options
from typing import List
import warnings

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

def plot_periodic_table(
    limiting_potential_dict: dict,
    show: bool = False,
    output_filename: str = None,
    width: int = 1050,
    cmap: str = "plasma",
    alpha: float = 0.65,
    extended: bool = True,
    periods_remove: List[int] = None,
    groups_remove: List[int] = None,
    cbar_height: float = None,
    cbar_standoff: int = 12,
    cbar_fontsize: int = 20,
    blank_color: str = "#c4c4c4",
    under_value: float = None,
    under_color: str = "#140F0E",
    over_value: float = None,
    over_color: str = "#140F0E",
    special_elements: List[str] = None,
    special_color: str = "#6F3023",
    cbar_location: tuple = (0, 0),  # location of colorbar
    cbar_range: dict = None,
) -> figure:

    """
    Plot a heatmap over the periodic table of elements.

    Parameters
    ----------
    filename : str
        Path to the .csv file containing the data to be plotted.
    show : str
        If True, the plot will be shown.
    output_filename : str
        If not None, the plot will be saved to the specified (.html) file.
    width : float
        Width of the plot.
    cmap : str
        plasma, inferno, viridis, magma, cividis, turbo
    alpha : float
        Alpha value (transparency).
    extended : bool
        If True, the lanthanoids and actinoids will be shown.
    periods_remove : List[int]
        Period numbers to be removed from the plot.
    groups_remove : List[int]
        Group numbers to be removed from the plot.
    log_scale : bool
        If True, the colorbar will be logarithmic.
    cbar_height : int
        Height of the colorbar.
    cbar_standoff : int
        Distance between the colorbar and the plot.
    cbar_fontsize : int
        Fontsize of the colorbar label.
    blank_color : str
        Hexadecimal color of the elements without data.
    under_value : float
        Values <= under_value will be colored with under_color.
    under_color : str
        Hexadecimal color to be used for the lower bound color.
    over_value : float
        Values >= over_value will be colored with over_color.
    under_color : str
        Hexadecial color to be used for the upper bound color.
    special_elements: List[str]
        List of elements to be colored with special_color.
    special_color: str
        Hexadecimal color to be used for the special elements.

    Returns
    -------
    figure
        Bokeh figure object.
    """

    options.mode.chained_assignment = None

    # Assign color palette based on input argument
    if cmap == "turbo":
        cmap = turbo
        bokeh_palette = "Turbo256"
    elif cmap == "coolwarm":
        cmap = coolwarm
        bokeh_palette = colorcet.coolwarm
        # bokeh_palette.reverse()  # reversed coolwarm colormap
    else:
        raise ValueError("Invalid color map.")

    # Define number of and groups
    period_label = ["1", "2", "3", "4", "5", "6", "7"]
    group_range = [str(x) for x in range(1, 19)]

    # Remove any groups or periods
    if groups_remove:
        for gr in groups_remove:
            gr = gr.strip()
            group_range.remove(str(gr))
    if periods_remove:
        for pr in periods_remove:
            pr = pr.strip()
            period_label.remove(str(pr))

    # Unpack limiting potential dict
    data_elements = [i.split("-")[-1] for i in limiting_potential_dict.keys()]
    data = list(limiting_potential_dict.values())


    period_label.append("blank")
    period_label.append("La")
    period_label.append("Ac")

    # Showing the lanthanoids and actinoids
    if extended:
        count = 0
        for i in range(56, 70):
            elements.period[i] = "La"
            elements.group[i] = str(count + 4)
            count += 1

        count = 0
        for i in range(88, 102):
            elements.period[i] = "Ac"
            elements.group[i] = str(count + 4)
            count += 1

    # Define matplotlib and bokeh color map
    color_mapper = LinearColorMapper(
        palette=bokeh_palette,
        low=min(data), high=max(data)
    )
    norm = Normalize(vmin=min(data), vmax=max(data))
    color_scale = ScalarMappable(norm=norm, cmap=cmap).to_rgba(data, alpha=None)

    # Set blank color
    color_list = [blank_color] * len(elements)

    # Compare elements in dataset with elements in periodic table
    for i, data_element in enumerate(data_elements):
        element_entry = elements.symbol[
            elements.symbol.str.lower() == data_element.lower()
        ]
        # detect if element entry is empty
        if element_entry.empty == False:
            element_index = element_entry.index[0]
        else:
            warnings.warn("Invalid chemical symbol: " + data_element)

        # determine color of each element
        if color_list[element_index] != blank_color:
            warnings.warn("Multiple entries for element " + data_element)
        elif under_value is not None and data[i] <= under_value:
            color_list[element_index] = under_color
        elif over_value is not None and data[i] >= over_value:
            color_list[element_index] = over_color
        else:
            color_list[element_index] = to_hex(color_scale[i])

    if special_elements:
        for k, v in elements["symbol"].iteritems():
            if v in special_elements:
                color_list[k] = special_color

    # Define figure properties for visualizing data
    source = ColumnDataSource(
        data=dict(
            group=[str(x) for x in elements["group"]],
            period=[str(y) for y in elements["period"]],
            sym=elements["symbol"],
            atomic_number=elements["atomic number"],
            type_color=color_list,
        )
    )

    # Plot the periodic table
    p = figure(x_range=group_range, y_range=list(reversed(period_label)), tools="save")
    p.width = width
    p.outline_line_color = None
    p.background_fill_color = None
    p.border_fill_color = None
    p.toolbar_location = "above"
    p.rect("group", "period", 0.9, 0.9, source=source, alpha=alpha, color="type_color")
    p.axis.visible = False
    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "left",
        "text_baseline": "middle",
    }
    x = dodge("group", -0.4, range=p.x_range)
    y = dodge("period", 0.3, range=p.y_range)
    p.text(
        x=x,
        y="period",
        text="sym",
        text_font_style="bold",
        text_font_size="16pt",
        **text_props,
    )
    p.text(x=x, y=y, text="atomic_number", text_font_size="11pt", **text_props)


    # Hide grid lines
    p.grid.grid_line_color = None


    # Add colorbar
    color_bar = ColorBar(
        color_mapper=color_mapper,
        border_line_color=None,
        location=cbar_location,  # position of the colorbar
        orientation="vertical",
        scale_alpha=alpha,

        # Label and tick settings
        major_label_text_font_size=f"{cbar_fontsize}pt",
        label_standoff=cbar_standoff,
        ticker=BasicTicker(desired_num_ticks=5),
        formatter=NumeralTickFormatter(format="0.0"),  # keep one decimal place
        )

    color_bar.major_tick_line_color = "black"  # set ticks to black
    color_bar.major_tick_out = 10  # set ticks outwards
    color_bar.major_tick_in = 0
    color_bar.major_tick_line_width = 2.5  # set ticks thickness

    ## Set colorbar value range
    if cbar_range is not None:
        color_mapper.low = cbar_range["low"]
        color_mapper.high = cbar_range["high"]

    ## Set colorbar height
    if cbar_height is not None:
        color_bar.height = cbar_height

    ## Put colorbar to the right of the plot
    p.add_layout(color_bar, "right")


    # Output and show plot
    if output_filename:
        export_png(p, filename=output_filename)

    if show:
        show_(p)