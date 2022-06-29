import shutil
from pathlib import Path

import numpy as np
from IPython.display import clear_output, display
from ipywidgets import Box, Button, Dropdown, Image, Label, Layout, Text, VBox

from .utils import extract_sample_images, prediction_dataframe


class JupyterViewer:
    """Graphical tool for viewing automatically classified IFCB images

    Parameters
    ----------
    prob : str, Path, list
        Path to a CSV-file containing class probabilities.
        A list of file paths can be provided only when mode is set to 'view'.
    raw_dir : str, Path
        Root directory of raw IFCB data.
    thresholds : float, str, Path
        Single value or path to a file with class specific thresholds values.
        Default is 0.0, meaning that every image is classified.
    mode : str
        Available options are: 'view', 'evaluate', 'label'.
    work_dir : str, Path
        Directory used to extract sample images and store progress files.
        By default it is created in ~/JupyterViewer.
    remove_extracted_images : bool
        Remove extracted images when the program exits. Default is True.
    remove_empty_work_dir : bool
        Remove empty `work_dir` when the program exits. Default is True.
    empty_name : str
        Name to use for unclassifiable images. Default is 'unclassifiable'.
    unsure_name : str
        Name to use for unsure images. Default is 'unsure'.

    Methods
    -------
    open()
    """

    def __init__(
        self,
        prob,
        raw_dir,
        thresholds=0.0,
        mode="view",
        work_dir=Path.home() / "JupyterViewer",
        remove_extracted_images=True,
        remove_empty_work_dir=True,
        empty_name="unclassifiable",
        unsure_name="unsure",
    ):

        self.work_dir = Path(work_dir)
        self.evaluate = mode == "evaluate"
        self.label = mode == "label"
        # This is not the best way to handle img sub dirs
        if isinstance(prob, list):
            if self.label or self.evaluate:
                raise ValueError("Labeling and evaluation is allowed one sample a time")
            self.img_dir = {}
            for csv in prob:
                sample = Path(csv).with_suffix("").stem
                self.img_dir[sample] = self.work_dir / "images" / sample
        else:
            self.sample = Path(prob).with_suffix("").stem
            self.img_dir = self.work_dir / "images" / self.sample

        self.df = prediction_dataframe(prob, thresholds)

        self.raw_dir = raw_dir
        self.select = self.label or self.evaluate
        self.remove_extracted_images = remove_extracted_images
        self.remove_empty_work_dir = remove_empty_work_dir
        self.labeled = {}
        self.moved = []
        self.empty = empty_name
        self.unsure = unsure_name
        self.extra_labels = []

        if self.select:
            # Update progress or start new
            if self.evaluate:
                self.sel_dir = self.work_dir / "evaluate"
                self.extra_labels.append("unsure")
            else:
                self.sel_dir = self.work_dir / "label"
            self.sel_log = self.sel_dir / f"{self.sample}.select.csv"
            self.moved_log = self.sel_dir / f"{self.sample}.copied.csv"
            if self.label and self.moved_log.is_file():
                i = 0
                with open(self.moved_log) as fh:
                    for i, line in enumerate(fh, start=1):
                        roi, label = line.strip().split(",")
                        self.moved.append(int(roi))
                print(
                    f"[INFO] Skipped {i} previously labeled images from: "
                    f"\n\t{self.moved_log}"
                )
            if self.sel_log.is_file():
                print(
                    "[INFO] Using previous selections for this "
                    f"sample from:\n\t{self.sel_log}"
                )
                print("[INFO] Remove it manually to start from scratch")
                with open(self.sel_log) as fh:
                    for line in fh:
                        roi, name = line.strip().split(",")
                        self.labeled[int(roi)] = name
            else:
                print(f"[INFO] Creating a new progress file in '{self.sel_log}'")
                Path.mkdir(self.sel_dir, parents=True, exist_ok=True)
            if self.label:
                self.extra_labels.append("New_Class")
                # Add any novel class labels found in work_dir
                for p in self.sel_dir.iterdir():
                    if p.is_dir():
                        cond_1 = p.name not in self.extra_labels
                        cond_2 = p.name not in self.df.columns[2:]
                        cond_3 = p.name != self.empty
                        if all((cond_1, cond_2, cond_3)):
                            self.extra_labels.append(p.name)

        self.item_layout = Layout(
            display="flex", flex_flow="column", align_items="center"
        )
        self.item_container_layout = Layout(
            dispay="flex", flex_flow="row wrap", align_items="baseline"
        )
        self.button_container_layout = Layout(
            display="flex ",
            flex_flow="row",
            justify_content="space-around",
            align_items="flex-end",
            margin="30px 0",
        )
        self.main_container_layout = Layout(
            display="flex", flex_flow="column", justify_content="space-between"
        )

    def open(
        self,
        per_page=42,
        start_page=1,
        sort_by="confidence",
        ascending=False,
        class_overview=False,
        class_filter=None,
        exclude_filtered=False,
        unsure_only=False,
    ):
        """Open the viewer

        Parameters
        ----------
        per_page : int
            Number of images to display per page.
        start_page : int
            Page number to start from.
        sort_by : str
            Available options are: 'confidence', 'class', 'roi'.
        ascending : bool
            Sort confidence or roi in an ascending order.
            By default they are sorted in a descending order.
        class_overview : bool
            Display each class on a separate page. If the number of
            images per class is bigger than `per_page`, a representative
            sample of them will be selected. This is useful for viewing
            overly abundant classes, but should not be used in evaluation.
        class_filter : str, list
            Filter images by classification. By default, only these classes
            are shown. Set `exclude_filtered` to True to reverse the behaviour.
        exclude_filtered : bool
            Exclude classes listed in `class_filter`. Default is False.
        unsure_only : bool
            Only show images labeled 'unsure', or whatever `unsure_name` is.
        """

        # Extract images only if they don't already exist
        # Multiple samples
        if isinstance(self.img_dir, dict):
            for sample, img_dir in self.img_dir.items():
                if not img_dir.is_dir() or len(list(img_dir.iterdir())) == 0:
                    print(f"[INFO] Extracting images for {sample}")
                    extract_sample_images(sample, self.raw_dir, img_dir, True)
        # One sample
        else:
            if not self.img_dir.is_dir() or len(list(self.img_dir.iterdir())) == 0:
                extract_sample_images(self.sample, self.raw_dir, self.img_dir, True)

        if sort_by not in ("confidence", "class", "roi"):
            sort_by = "roi"

        # Filter images by classification
        if class_filter:
            if isinstance(class_filter, str):
                class_filter = [class_filter]
            for name in class_filter:
                assert name in self.df.columns[2:], f"Unknown class '{name}'"
            if exclude_filtered:
                df = self.df[~self.df["prediction"].isin(class_filter)]
            else:
                df = self.df[self.df["prediction"].isin(class_filter)]
            if len(df) <= 0:
                print("[INFO] No predictions to show")
                return
        else:
            df = self.df

        if self.label:
            # Don't show previously labeled and moved images
            df = df.drop(self.moved)

        if self.evaluate and unsure_only:
            # Only show images labeled 'unsure'
            unsure_roi = [
                roi for roi, label in self.labeled.items() if label == self.unsure
            ]
            df = df[df.index.isin(unsure_roi)]
            if len(df) <= 0:
                print("[INFO] No predictions to show")
                return

        # Divide images to pages
        self.pages = []
        if sort_by == "class" or class_overview:
            # Group data frame by classes
            for name, group in df.sort_values("prediction").groupby("prediction"):
                if len(group) < 1:
                    continue
                group_indeces = group.sort_values(name, ascending=ascending).index
                num_preds = len(group_indeces)
                if class_overview:
                    # Choose random subset of indeces from inside each
                    # prediction group, across different confidence values
                    # Show at max, 'per_page' number of predictions per class
                    num_preds_to_display = min(num_preds, per_page)
                    subset = np.linspace(
                        0, num_preds - 1, num=num_preds_to_display, dtype=int
                    )
                    # Create one page for each prediction group
                    self.pages.append(group_indeces[subset])
                else:
                    # Create as many pages as it takes for this class
                    self.pages.extend(
                        [
                            group_indeces[i : i + per_page]
                            for i in range(0, num_preds, per_page)
                        ]
                    )
        else:
            # Sort images by confidence
            if sort_by == "confidence":
                df["confidence"] = df.apply(lambda row: row[row["prediction"]], axis=1)
                df.sort_values("confidence", ascending=ascending, inplace=True)
                df.drop("confidence", axis=1, inplace=True)
            # Sort all predictions by roi number (index)
            else:
                df.sort_index(ascending=ascending, inplace=True)
            self.unlabeled = df.index.tolist()
            # Divide indeces to equal sized pages
            for i in range(0, len(self.unlabeled), per_page):
                self.pages.append(self.unlabeled[i : i + per_page])

        self.current_page = start_page - 1
        self._show_current_page()

    def _show_current_page(self, *args):
        clear_output()
        if not self.pages:
            print("[INFO] No predictions to show")
            return
        next_disabled = True if self.current_page == len(self.pages) - 1 else False
        back_disabled = True if self.current_page == 0 else False
        unlabeled = self.pages[self.current_page]
        items = [self._new_item(roi) for roi in unlabeled]
        next_btn = Button(
            description=">",
            button_style="info",
            disabled=next_disabled,
            tooltip="Show next page",
        )
        next_btn.on_click(self._next_button_handler)
        back_btn = Button(
            description="<",
            button_style="info",
            disabled=back_disabled,
            tooltip="Show previous page",
        )
        back_btn.on_click(self._back_button_handler)
        end_tooltip = "Close viewer"
        if self.remove_extracted_images:
            end_tooltip += " and remove extracted images"
        end_btn = Button(
            description="Close", button_style="warning", tooltip=end_tooltip
        )
        end_btn.on_click(self._end_button_handler)
        item_container = Box(children=items, layout=self.item_container_layout)
        button_container = Box(
            children=[end_btn, back_btn, next_btn], layout=self.button_container_layout
        )
        main_container = Box(
            children=[item_container, button_container],
            layout=self.main_container_layout,
        )
        display(main_container)
        print(f"Page {self.current_page+1} / {len(self.pages)}")
        if self.select:
            self._log_selections()

    def _new_item(self, roi):
        row = self.df.loc[roi]
        prediction = row["prediction"] if row["classified"] else self.empty
        confidence = row[row["prediction"]]
        probs = row[2:].sort_values(ascending=False)

        if isinstance(self.img_dir, dict):
            # Extract multi-index
            sample, roi_num = roi
            try:
                with open(
                    self.img_dir[sample] / f"{sample}_{roi_num:05}.png", "rb"
                ) as fh:
                    img = Image(value=fh.read(), format="png")
            except FileNotFoundError:
                print("[ERROR] Images have likely been moved due to labeling")
                print(f"[INFO] Try deleting {self.work_dir}/images")
                raise
            # Label to display below each image
            label = Label(f"{confidence:.5f}")
        else:
            try:
                with open(self.img_dir / f"{self.sample}_{roi:05}.png", "rb") as fh:
                    img = Image(value=fh.read(), format="png")
            except FileNotFoundError:
                print("[ERROR] Images have likely been moved due to labeling.")
                print(f"[INFO] Try deleting {self.work_dir}/images")
                raise
            label = Label(f"{roi} - {confidence:.5f}")

        if self.select:
            if roi in self.labeled:
                # This image has been seen before
                previous = self.labeled[roi]
                if previous != prediction:
                    # Previous selection was a correction
                    # Add check mark to distinguish this
                    label.value = "\N{White Heavy Check Mark} " + label.value
                    prediction = previous
            else:
                self.labeled[roi] = prediction
        # Set dropdown selection to empty initially
        options = [""]
        predicted_option = "" if prediction != self.unsure else self.unsure
        for name, prob in probs.items():
            option = f"{prob:.5f} - {name}"
            options.append(option)
            if name == prediction:
                # Set dropdown selection to predicted class
                predicted_option = option
        if self.extra_labels:
            options.insert(1, *self.extra_labels)
            # Select previous selection if it is in extra labels
            if not predicted_option:
                for name in self.extra_labels:
                    if name == prediction:
                        predicted_option = name
        dropdown = Dropdown(options=options, value=predicted_option)
        if self.select:
            # Add listener to dropdown menu
            dropdown.observe(self._dropdown_handler(roi), names="value")
        item = Box(children=[img, label, dropdown], layout=self.item_layout)
        return item

    def _cleanup(self):
        if not self.remove_extracted_images:
            return
        if isinstance(self.img_dir, dict):
            for img_dir in self.img_dir.values():
                shutil.rmtree(img_dir)
                print(f"[INFO] Removed image directory: {img_dir}")
        else:
            shutil.rmtree(self.img_dir)
            print(f"[INFO] Removed image directory: {self.img_dir}")
        # Remove empty 'images' dir
        try:
            self.img_dir.parent.rmdir()
            print(f"[INFO] Removed empty directory: {self.img_dir.parent}")
        except OSError:
            pass
        # Remove empty work dir
        if self.remove_empty_work_dir:
            try:
                self.work_dir.rmdir()
                print(f"[INFO] Removed empty directory: {self.work_dir}")
            except OSError:
                pass

    def _next_button_handler(self, button):
        self.current_page += 1
        self._show_current_page()

    def _back_button_handler(self, button):
        self.current_page -= 1
        self._show_current_page()

    def _end_button_handler(self, button):
        clear_output()
        if self.select:
            self._log_selections()
        if not self.label or not self.labeled:
            self._cleanup()
            return
        if self.label:
            num_labeled = len(
                [name for name in self.labeled.values() if name != self.empty]
            )
            print(f"[INFO] Total labeled images: {num_labeled}")
            dest = Text(
                value=str(self.sel_dir),
                description="Copy to:",
                tooltip="Each label gets a sub-directory automatically",
            )
            move_btn = Button(
                description="Accept",
                button_style="success",
                tooltip="Copy labeled images and exit",
            )
            move_btn.dest = dest
            move_btn.on_click(self._move_button_handler)
            back_btn = Button(
                description="Back", button_style="info", tooltip="Back to viewer"
            )
            back_btn.on_click(self._show_current_page)
            quit_btn = Button(
                description="Quit",
                button_style="danger",
                tooltip="Exit without copying labeled images",
            )
            quit_btn.on_click(self._quit_button_handler)
            buttons = Box(
                children=[quit_btn, back_btn, move_btn],
                layout=Layout(margin="30px 0 10px 0"),
            )
            display(VBox([dest, buttons]))

    def _move_button_handler(self, button):
        clear_output()
        dest_dir = Path(button.dest.value)
        copied = []
        with open(self.moved_log, "a") as fh:
            for roi, label in self.labeled.items():
                if not label or label == self.empty:
                    # Don't copy unlabeled images
                    continue
                img_name = f"{self.sample}_{roi:05}.png"
                src = self.img_dir / img_name
                dst = dest_dir / label / img_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                fh.write(f"{roi},{label}\n")
                print(f"[INFO] Copied {roi} to {dst}")
                copied.append(roi)
        # Remove copied items from labeled, add them to moved
        for roi in copied:
            self.labeled.pop(roi)
            self.moved.append(roi)
        # Write updated labeled to file
        self._log_selections()
        self._cleanup()

    def _quit_button_handler(self, button):
        clear_output()
        self._cleanup()

    def _dropdown_handler(self, roi):
        def handler(change):
            self.labeled[roi] = change.new.split(" - ")[-1]

        return handler

    def _log_selections(self):
        data = ""
        for roi, selection in self.labeled.items():
            if not selection:
                # Set empty selection to a string
                selection = self.empty
            data += f"{roi},{selection}\n"
        with open(self.sel_log, "w") as fh:
            fh.write(data)
