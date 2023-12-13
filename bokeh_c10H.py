# %%
#
#  Run the following command to see the results:
# conda activate peerannot;
# bokeh serve --show bokeh_c10H.py --port 5007;
#
#
# see http://14.99.188.242:8080/jspui/bitstream/123456789/14487/1/Hands-On%20Data%20Visualization%20with%20Bokeh%20Interactive%20Web%20Plotting%20for%20Python%20Using%20Bokeh%20by%20Kevin%20Jolly%20%28z-lib.org%29.pdf page 149

from numpy.random import random, normal
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import (
    Select,
    BoxSelectTool,
    LassoSelectTool,
    ColumnDataSource,
    Slider,
    Div,
)
from bokeh.io import curdoc
from bokeh.models.glyphs import Circle
from bokeh.palettes import Category10
from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    Select,
    Legend,
)
from bokeh.plotting import figure, output_file
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import torchvision
import os
from peerannot.models import NaiveSoft
from scipy.special import entr
import json
from scipy import interpolate

# %%


def aum_from_logits(data, burn=0):
    tasks = {"sample_id": [], "AUM_yang": [], "AUM_pleiss": []}
    for task in data.sample_id.unique():
        tmp = data[data.sample_id == task]
        y = tmp.target.iloc[0]
        target_values = tmp.target_values.values[burn:]
        logits = tmp.values[burn:, -10:]
        llogits = np.copy(logits)
        _ = np.put_along_axis(logits, logits.argmax(1).reshape(-1, 1), float("-inf"), 1)
        masked_logits = logits
        other_logit_values, other_logit_index = masked_logits.max(
            1
        ), masked_logits.argmax(1)
        other_logit_values = other_logit_values.squeeze()
        other_logit_index = other_logit_index.squeeze()
        margin_values_yang = (target_values - other_logit_values).tolist()
        _ = np.put_along_axis(
            llogits, np.repeat(y, len(tmp)).reshape(-1, 1), float("-inf"), 1
        )
        masked_logits = llogits
        other_logit_values, other_logit_index = masked_logits.max(
            1
        ), masked_logits.argmax(1)
        other_logit_values = other_logit_values.squeeze()
        other_logit_index = other_logit_index.squeeze()
        margin_values_pleiss = (target_values - other_logit_values).mean()

        tasks["sample_id"].append(task)
        tasks["AUM_yang"].append(np.mean(margin_values_yang))
        tasks["AUM_pleiss"].append(np.mean(margin_values_pleiss))
    df = pd.DataFrame(tasks)
    return df


# %%
# Create the figure arguments
common_fig_kwargs = {
    "width": 850,
    "height": 400,
    "tools": ("pan, wheel_zoom, reset, tap, box_select, box_zoom,lasso_select"),
}

common_line_kwargs = {
    "line_width": 4,
    "line_dash": "dashed",
    "alpha": 0.9,
}

common_circle_kwargs = {
    "name": "circle",
    "size": 10,
}

tooltips_waum = """
<div>
    <div>
        <img src='@image' height="310" width="310" style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>WAUM:</span>
        <span style='font-size: 18px'>@waum</span>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Sample_id:</span>
        <span style='font-size: 18px'>@sample_id</span>
    </div>
        <div>
        <span style='font-size: 16px; color: #224499'>Label:</span>
        <span style='font-size: 18px'>@label</span>
    </div>
</div>
"""

tooltips_aum = """
<div>
    <div>
        <img src='@image' height="310" width="310" style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>AUM:</span>
        <span style='font-size: 18px'>@aums_vanilla</span>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Sample_id:</span>
        <span style='font-size: 18px'>@sample_id</span>
    </div>
        <div>
        <span style='font-size: 16px; color: #224499'>Label:</span>
        <span style='font-size: 18px'>@label</span>
    </div>
</div>
"""

# %%

with open("./data/cifar10h_p_0_spam_0_all_workers.json", "r") as votes:
    labels_json = json.load(votes)
labels_json = dict(
    sorted(
        {
            int(taskid): {
                int(worker): int(lab) for worker, lab in labels_json[taskid].items()
            }
            for taskid in labels_json.keys()
            if int(taskid) < 9500
        }.items()
    )
)


output_file("toolbar.html", title="CIFAR10-H")
FONT_SIZE = 12
ALL_LABELS = True
INIT_SAMPLE = 982

# %%
dirfile = os.getcwd()
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
    ),
)

batch_size = 64


aums = pd.read_csv(os.path.join(dirfile, "waums_c10h_red_yang.csv"))
aums_pleiss = pd.read_csv(os.path.join(dirfile, "waum_C10H_pleiss.csv"))
aums["waum_pleiss"] = aums_pleiss["waum"]
aums["waum_yang"] = aums["waum"]

resnet = 18
full_logits = pd.read_parquet(
    os.path.join(dirfile, "results", f"resnet{resnet}", f"logits_full_{resnet}.parquet")
)
aum_df = aum_from_logits(full_logits)
# %%
dict_classes = trainset.class_to_idx
classes = list(dict_classes.keys())
inv_classes = dict(zip(dict_classes.values(), dict_classes.keys()))
# %%
# subsampling for speed testing (n_sub = 9500 for all data)
n_sub = 9500
aums = aums[:n_sub]
aums_vanilla = aum_df[:n_sub]
soft = NaiveSoft(labels_json, n_classes=10)
yp = soft.get_probas()
entropy = entr(yp).sum(1)
idx_correct = aums["sample_id"].tolist()
entropy = entropy[idx_correct[:n_sub]]


# %%

mean_cifar = np.array([0.0, 0.0, 0.0])
std_cifar = np.array([1.0, 1.0, 1.0])

inv_normalize = torchvision.transforms.Normalize(
    mean=mean_cifar / std_cifar, std=1.0 / std_cifar
)


labels = []
images = []
for idx, data in enumerate(trainset):
    labels.extend([inv_classes[data[1]]])
    image = 255 * (np.transpose(inv_normalize(data[0]).numpy(), (1, 2, 0)))
    images.extend([image.astype("uint8")])

# labels, inputs = np.asarray(y_labels), np.asarray(inputs_)

print(len(labels), len(images))


def embeddable_image(data):
    image = Image.fromarray(data, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="jpeg")
    for_encoding = buffer.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(for_encoding).decode()


# %%
# select in aums lines with the label menu.value
AUM_type = "yang"
# WAUM_type = "waum_" + AUM_type
labels_shuffled = [labels[i] for i in idx_correct]
images_shuffled = [images[i] for i in idx_correct]
aums["label"] = labels_shuffled
aums["image"] = list(map(embeddable_image, images_shuffled))
aums["entropy"] = entropy

# # WARNING substitute time to entropy
# fname_csv = 'data/cifar10h-raw.csv'
# df_cifar10h = pd.read_csv(fname_csv, na_values='-99999')
# df_cifar10h.dropna(inplace=True)
# # in df_cifar10h trasnform column cifar10-test-test-idx in int
# df_cifar10h['cifar10_test_test_idx'] = df_cifar10h['cifar10_test_test_idx'].astype(
#     int)
# # sort df_cifar10h by 'cifar10_test_test_idx' columns
# df_cifar10h = df_cifar10h.sort_values(by=['cifar10_test_test_idx'])
# images_perf = df_cifar10h.groupby('image_filename')
# # idx_correct[:n_sub]
# # df_cifar10h = df_cifar10h[:n_sub]
# aums["entropy"] = images_perf['reaction_time'].median().values[idx_correct[:n_sub]]


aums["aums_vanilla"] = aums_vanilla[f"AUM_{AUM_type}"].tolist()
aums["sizes_selected"] = 20

# %% Display part


data_points_df = ColumnDataSource(data=aums[aums["label"] == classes[0]])
# data_points_vanilla_df = ColumnDataSource(
# data=aums_vanilla[aums_vanilla["label"] == classes[0]])


# Create the WAUM plot
plot_waum = figure(
    **common_fig_kwargs,
    title="WAUMs (using crowdsrouced labels)",
    x_range=(aums["waum"].min() - 0.04, aums["waum"].max() + 0.04),
    y_range=(aums["entropy"].min() - 0.1, aums["entropy"].max() + 0.1),
)
plot_waum.xaxis.axis_label = "WAUM"
plot_waum.yaxis.axis_label = "Entropy"
plot_waum.title.text_font_size = "15pt"
plot_waum.xaxis.axis_label_text_font_size = "15pt"
plot_waum.yaxis.axis_label_text_font_size = "15pt"


global_slider = Slider(
    start=0.01, end=1, value=0.01, step=0.01, title="Global quantile"
)
local_slider = Slider(start=0.01, end=1, value=0.01, step=0.01, title="Local quantile")

alpha_global = global_slider.value
alpha_local = local_slider.value
q_global_waum = np.quantile(aums["waum"], alpha_global)
q_local_waum = np.quantile(data_points_df.data["waum"], alpha_local)

lower = plot_waum.y_range.start
upper = plot_waum.y_range.end
line_global_waum = plot_waum.line(
    [q_global_waum, q_global_waum],
    [lower, upper],
    **common_line_kwargs,
    line_color="black",
    legend_label="Global quantile",
)

line_local_waum = plot_waum.line(
    [q_local_waum, q_local_waum],
    [lower, upper],
    **common_line_kwargs,
    line_color="orange",
    legend_label="Local quantile",
)

selection_glyph = Circle(
    line_alpha=0.4,
    fill_alpha=0.4,
    line_width=3,
    size="sizes_selected",
    fill_color="#FF0000",
)

nonselection_glyph = Circle(line_alpha=0.2, fill_alpha=0.2, fill_color="#3288bd")
c1 = plot_waum.circle(
    x="waum",
    y="entropy",
    **common_circle_kwargs,
    source=data_points_df,
)
c1.selection_glyph = selection_glyph
c1.nonselection_glyph = nonselection_glyph

plot_waum.legend.click_policy = "hide"

plot_waum.add_tools(HoverTool(name="circle", tooltips=tooltips_waum))

# %%
# Create the AUM plot
plot_aum = figure(
    title='AUMs (using so called "true labels")',
    **common_fig_kwargs,
    x_range=(
        aums["aums_vanilla"].min() - 0.4,
        aums["aums_vanilla"].max() + 0.4,
    ),
    y_range=(aums["entropy"].min() - 0.1, aums["entropy"].max() + 0.1),
)

plot_aum.width = 950
c2 = plot_aum.circle(
    x="aums_vanilla",
    y="entropy",
    **common_circle_kwargs,
    source=data_points_df,
)

c2.selection_glyph = selection_glyph
c2.nonselection_glyph = nonselection_glyph

plot_aum.xaxis.axis_label = "AUM"
plot_aum.yaxis.axis_label = "Entropy"
plot_aum.title.text_font_size = "15pt"
plot_aum.xaxis.axis_label_text_font_size = "15pt"
plot_aum.yaxis.axis_label_text_font_size = "15pt"

q_global_aum = np.quantile(aums_vanilla[f"AUM_{AUM_type}"], alpha_global)
q_local_aum = np.quantile(data_points_df.data["aums_vanilla"], alpha_local)

line_global_aum = plot_aum.line(
    [q_global_aum, q_global_aum],
    [plot_aum.y_range.start, plot_aum.y_range.end],
    **common_line_kwargs,
    line_color="black",
    legend_label="Global quantile",
)

line_local_aum = plot_aum.line(
    [q_local_aum, q_local_aum],
    [plot_aum.y_range.start, plot_aum.y_range.end],
    **common_line_kwargs,
    line_color="orange",
    legend_label="Local quantile",
)

plot_aum.legend.location = "top_left"
plot_aum.legend.click_policy = "hide"
plot_aum.add_tools(HoverTool(name="circle", tooltips=tooltips_aum))

# %%
# Create the barplot
plot_bar = figure(
    **common_fig_kwargs,
    x_range=classes,
    title="Crowdsourced labels distribution",
)

plot_bar.yaxis.axis_label = "Frequency"
plot_bar.xaxis.major_label_text_font_size = "15px"
plot_bar.xaxis.axis_label_text_font_size = "15pt"
plot_bar.yaxis.axis_label_text_font_size = "15pt"
plot_bar.title.text_font_size = "15pt"


data_points_df2 = ColumnDataSource(
    data={
        "Labels": classes,
        "Frequency": yp[INIT_SAMPLE],
        "colors": list(Category10[10]),
    }
)

plot_bar.vbar(
    x="Labels",
    top="Frequency",
    source=data_points_df2,
    alpha=1,
    color="colors",
)
plot_bar.xaxis.major_label_orientation = 0.8

# %%
# Create the logits plot

plot_logi = figure(**common_fig_kwargs, title="Logit evolution while training")
plot_logi.width = 950

plot_logi.add_layout(Legend(), "right")
data_lines_structure = ColumnDataSource(
    data={
        "line_width": [7, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        "line_alpha": [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }
)

col_names = full_logits.columns[4:]

aaaaa = full_logits[full_logits.sample_id == INIT_SAMPLE]
true_class = aaaaa["target"].iloc[0]

x_old = aaaaa["num_measurements"].values
x_new = np.linspace(np.min(x_old), np.max(x_old), len(aaaaa.num_measurements) * 10)
aaaaa_super_res = pd.DataFrame()
aaaaa_super_res["num_measurements"] = x_new

for col_name in col_names:
    f_interpol = interpolate.interp1d(x_old, aaaaa[col_name])
    aaaaa_super_res[col_name] = f_interpol(x_new)
f_interpol_aum = interpolate.interp1d(x_old, aaaaa["target_values"])
aaaaa_super_res["target_values"] = f_interpol_aum(x_new)

if AUM_type == "yang":
    aaaaa_super_res["second_largest"] = aaaaa_super_res[col_names.values].apply(
        lambda a: np.partition(a, -2)[-2], axis=1
    )
else:
    aaaaa_super_res["second_largest"] = aaaaa_super_res[
        col_names.drop(col_names[true_class]).values
    ].apply(np.max, axis=1)

aaaaa_super_res["second_largest_blue"] = aaaaa_super_res[
    ["second_largest", "target_values"]
].apply(np.max, axis=1)
aaaaa_super_res["second_largest_red"] = aaaaa_super_res[
    ["second_largest", "target_values"]
].apply(np.min, axis=1)


data_pts_vanilla_df = ColumnDataSource(aaaaa_super_res)
my_lines = dict()

for i, col_name in enumerate(col_names):
    my_lines[i] = plot_logi.line(
        data_pts_vanilla_df.data["num_measurements"],
        data_pts_vanilla_df.data[col_name],
        color=Category10[10][i],
        line_width=data_lines_structure.data["line_width"][i],
        line_alpha=data_lines_structure.data["line_alpha"][i],
        legend_label=classes[i],
    )

varea1 = plot_logi.varea(
    x=data_pts_vanilla_df.data["num_measurements"],
    y1=data_pts_vanilla_df.data[col_names[true_class]],
    y2=data_pts_vanilla_df.data["second_largest_blue"],
    color="red",
    alpha=0.2,
)
varea2 = plot_logi.varea(
    x=data_pts_vanilla_df.data["num_measurements"],
    y1=data_pts_vanilla_df.data[col_names[true_class]],
    y2=data_pts_vanilla_df.data["second_largest_red"],
    color="blue",
    alpha=0.2,
)
plot_logi.legend.click_policy = "hide"

plot_logi.yaxis.axis_label = "Logits"
plot_logi.xaxis.axis_label = "Epochs"
plot_logi.legend.location = (0, 14)

plot_logi.xaxis.axis_label_text_font_size = "15pt"
plot_logi.yaxis.axis_label_text_font_size = "15pt"
plot_logi.title.text_font_size = "15pt"


# %%
menu = Select(options=classes, value=classes[0], title="Label")
aum_menu = Select(
    options=["Yang", "Pleiss"],
    value="Pleiss",
    title="AUM margin",
)


selected_points = [INIT_SAMPLE]


def update_global(attr, old, new):
    alpha_global = global_slider.value
    q_global_aum = np.quantile(aums_vanilla[f"AUM_{AUM_type}"], alpha_global)
    q_global_waum = np.quantile(aums["waum"], alpha_global)
    line_global_aum.data_source.data["x"] = [q_global_aum, q_global_aum]
    line_global_waum.data_source.data["x"] = [q_global_waum, q_global_waum]


def update_local(attr, old, new):
    alpha_local = local_slider.value
    q_local_aum = np.quantile(data_points_df.data["aums_vanilla"], alpha_local)
    q_local_waum = np.quantile(data_points_df.data["waum"], alpha_local)
    line_local_aum.data_source.data["x"] = [q_local_aum, q_local_aum]
    line_local_waum.data_source.data["x"] = [q_local_waum, q_local_waum]


def callback_aum_menu(attr, old, new):
    AUM_type = aum_menu.value.lower()
    aums["aums_vanilla"] = aums_vanilla[f"AUM_{AUM_type}"].tolist()
    data_points_df.data["waum"] = data_points_df.data[f"waum_{AUM_type}"]
    data_points_df.data["aums_vanilla"] = aums[aums["label"] == menu.value][
        "aums_vanilla"
    ]

    update_global(attr, old, new)
    update_local(attr, old, new)

    aaaaa = full_logits[full_logits.sample_id == selected_points[-1]]
    true_class = aaaaa["target"].iloc[0]

    x_old = aaaaa["num_measurements"].values
    x_new = np.linspace(np.min(x_old), np.max(x_old), len(aaaaa.num_measurements) * 10)
    aaaaa_super_res = pd.DataFrame()
    aaaaa_super_res["num_measurements"] = x_new

    for col_name in col_names:
        f_interpol = interpolate.interp1d(x_old, aaaaa[col_name])
        aaaaa_super_res[col_name] = f_interpol(x_new)
    f_interpol_aum = interpolate.interp1d(x_old, aaaaa["target_values"])
    aaaaa_super_res["target_values"] = f_interpol_aum(x_new)

    if aum_menu.value.lower() == "yang":
        aaaaa_super_res["second_largest"] = aaaaa_super_res[col_names.values].apply(
            lambda a: np.partition(a, -2)[-2], axis=1
        )
    else:
        aaaaa_super_res["second_largest"] = aaaaa_super_res[
            col_names.drop(col_names[true_class]).values
        ].apply(np.max, axis=1)

    aaaaa_super_res["second_largest_blue"] = aaaaa_super_res[
        ["second_largest", "target_values"]
    ].apply(np.max, axis=1)
    aaaaa_super_res["second_largest_red"] = aaaaa_super_res[
        ["second_largest", "target_values"]
    ].apply(np.min, axis=1)

    data_pts_vanilla_df.data = aaaaa_super_res

    varea1.data_source.data["x"] = data_pts_vanilla_df.data["num_measurements"]
    varea1.data_source.data["y1"] = data_pts_vanilla_df.data[col_names[true_class]]
    varea1.data_source.data["y2"] = data_pts_vanilla_df.data["second_largest_blue"]
    varea2.data_source.data["x"] = data_pts_vanilla_df.data["num_measurements"]
    varea2.data_source.data["y1"] = data_pts_vanilla_df.data[col_names[true_class]]
    varea2.data_source.data["y2"] = data_pts_vanilla_df.data["second_largest_red"]

    for i, col_name in enumerate(col_names):
        col_name = "logit{}".format(i)
        my_lines[i].data_source.data["y"] = data_pts_vanilla_df.data[col_name]
        if i == true_class:
            line_width = 7
            line_alpha = 1
        else:
            line_width = 3
            line_alpha = 0.5
        my_lines[i].glyph.line_width = line_width
        my_lines[i].glyph.line_alpha = line_alpha


def callback(attr, old, new):
    idx = aums["label"] == menu.value
    data_points_df.data = aums[idx]
    update_global(attr, old, new)
    update_local(attr, old, new)


def my_tap_handler(attr, old, new):
    index = data_points_df.data["sample_id"][new[0]]
    selected_points.append(index)
    data_points_df2.data = {
        "Labels": classes,
        "Frequency": yp[index],
        "colors": list(Category10[10]),
    }

    aaaaa = full_logits[full_logits.sample_id == index]
    true_class = aaaaa["target"].iloc[0]

    x_old = aaaaa["num_measurements"].values
    x_new = np.linspace(np.min(x_old), np.max(x_old), len(aaaaa.num_measurements) * 10)
    aaaaa_super_res = pd.DataFrame()
    aaaaa_super_res["num_measurements"] = x_new

    for col_name in col_names:
        f_interpol = interpolate.interp1d(x_old, aaaaa[col_name])
        aaaaa_super_res[col_name] = f_interpol(x_new)
    f_interpol_aum = interpolate.interp1d(x_old, aaaaa["target_values"])
    aaaaa_super_res["target_values"] = f_interpol_aum(x_new)

    if aum_menu.value.lower() == "yang":
        aaaaa_super_res["second_largest"] = aaaaa_super_res[col_names.values].apply(
            lambda a: np.partition(a, -2)[-2], axis=1
        )
    else:
        aaaaa_super_res["second_largest"] = aaaaa_super_res[
            col_names.drop(col_names[true_class]).values
        ].apply(np.max, axis=1)

    aaaaa_super_res["second_largest_blue"] = aaaaa_super_res[
        ["second_largest", "target_values"]
    ].apply(np.max, axis=1)
    aaaaa_super_res["second_largest_red"] = aaaaa_super_res[
        ["second_largest", "target_values"]
    ].apply(np.min, axis=1)

    data_pts_vanilla_df.data = aaaaa_super_res

    varea1.data_source.data["x"] = data_pts_vanilla_df.data["num_measurements"]
    varea1.data_source.data["y1"] = data_pts_vanilla_df.data[col_names[true_class]]
    varea1.data_source.data["y2"] = data_pts_vanilla_df.data["second_largest_blue"]
    varea2.data_source.data["x"] = data_pts_vanilla_df.data["num_measurements"]
    varea2.data_source.data["y1"] = data_pts_vanilla_df.data[col_names[true_class]]
    varea2.data_source.data["y2"] = data_pts_vanilla_df.data["second_largest_red"]

    for i, col_name in enumerate(col_names):
        col_name = "logit{}".format(i)
        my_lines[i].data_source.data["y"] = data_pts_vanilla_df.data[col_name]
        if i == true_class:
            line_width = 7
            line_alpha = 1
        else:
            line_width = 3
            line_alpha = 0.5
        my_lines[i].glyph.line_width = line_width
        my_lines[i].glyph.line_alpha = line_alpha


global_slider.on_change("value", update_global)
local_slider.on_change("value", update_local)

aum_menu.on_change("value", callback_aum_menu)
menu.on_change("value", callback)
data_points_df.selected.on_change("indices", my_tap_handler)
data_pts_vanilla_df.selected.on_change("indices", my_tap_handler)

select_overlay = plot_waum.select_one(BoxSelectTool).overlay
select_overlay.fill_color = "firebrick"
select_overlay.line_color = None

plot_waum.select_one(LassoSelectTool).overlay.line_dash = [10, 10]
title_html = Div(text='<h1 style="text-align: center">CIFAR10H AUMs and WAUMs</h1>')
layout = gridplot(
    [
        [title_html, menu],
        [aum_menu, global_slider],
        [None, local_slider],
        [plot_aum, plot_waum],
        [plot_logi, plot_bar],
    ],
)
# Add the layout to the application
curdoc().add_root(layout)
