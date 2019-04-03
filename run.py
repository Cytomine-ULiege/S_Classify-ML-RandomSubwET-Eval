# -*- coding: utf-8 -*-

# /**
# * Copyright (c) 2009-2018.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

import os
import numpy as np
from pathlib import Path
from sklearn.externals import joblib
from cytomine.models import *
from cytomine import CytomineJob
from cytomine.utilities.software import setup_classify, parse_domain_list, stringify
from pyxit import build_models
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit, GroupKFold, StratifiedKFold
from sklearn.utils import check_random_state

__author__          = "Mormont Romain <r.mormont@uliege.be>"
__copyright__       = "Copyright 2010-2018 University of Liège, Belgium, http://www.cytomine.org/"


def window_indexes(n, idx, count):
    increment = np.tile(np.arange(count), (idx.shape[0], 1)).flatten()
    starts = np.repeat(np.arange(n)[idx], count)
    return starts * count + increment


def array2str(arr):
    return "[" + (", ".join(["{}".format(v) for v in arr])) + "]"


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # annotation filtering
        cj.logger.info(str(cj.parameters))

        cj.job.update(progress=1, statuscomment="Preparing execution (creating folders,...).")
        base_path, downloaded = setup_classify(
            args=cj.parameters, logger=cj.job_logger(1, 40),
            dest_pattern=os.path.join("{term}", "{image}_{id}.png"),
            root_path=Path.home(), set_folder="train", showTerm=True
        )

        x = np.array([f for annotation in downloaded for f in annotation.filenames])
        y = np.array([int(os.path.basename(os.path.dirname(filepath))) for filepath in x])

        # transform classes
        cj.job.update(progress=50, statusComment="Transform classes...")
        classes = parse_domain_list(cj.parameters.cytomine_id_terms)
        positive_classes = parse_domain_list(cj.parameters.cytomine_positive_terms)
        classes = np.array(classes) if len(classes) > 0 else np.unique(y)
        n_classes = classes.shape[0]

        # filter unwanted terms
        cj.logger.info("Size before filtering:")
        cj.logger.info(" - x: {}".format(x.shape))
        cj.logger.info(" - y: {}".format(y.shape))
        keep = np.in1d(y, classes)
        x, y = x[keep], y[keep]
        cj.logger.info("Size after filtering:")
        cj.logger.info(" - x: {}".format(x.shape))
        cj.logger.info(" - y: {}".format(y.shape))

        labels = np.array([int(os.path.basename(f).split("_", 1)[0]) for f in x])

        if cj.parameters.cytomine_binary:
            cj.logger.info("Will be training on 2 classes ({} classes before binarization).".format(n_classes))
            y = np.in1d(y, positive_classes).astype(np.int)
        else:
            cj.logger.info("Will be training on {} classes.".format(n_classes))
            y = np.searchsorted(classes, y)

        # build model
        random_state = check_random_state(cj.parameters.seed)
        cj.job.update(progress=55, statusComment="Build model...")
        _, pyxit = build_models(
            n_subwindows=cj.parameters.pyxit_n_subwindows,
            min_size=cj.parameters.pyxit_min_size,
            max_size=cj.parameters.pyxit_max_size,
            target_width=cj.parameters.pyxit_target_width,
            target_height=cj.parameters.pyxit_target_height,
            interpolation=cj.parameters.pyxit_interpolation,
            transpose=cj.parameters.pyxit_transpose,
            colorspace=cj.parameters.pyxit_colorspace,
            fixed_size=cj.parameters.pyxit_fixed_size,
            verbose=int(cj.logger.level == 10),
            create_svm=cj.parameters.svm,
            C=cj.parameters.svm_c,
            random_state=random_state,
            n_estimators=cj.parameters.forest_n_estimators,
            min_samples_split=cj.parameters.forest_min_samples_split,
            max_features=cj.parameters.forest_max_features,
            n_jobs=cj.parameters.n_jobs
        )

        cj.job.update(progress=60, statusComment="Start cross-validation...")
        n_splits = cj.parameters.eval_k
        cv = ShuffleSplit(n_splits, test_size=cj.parameters.eval_test_fraction)
        if cj.parameters.folds == "group":
            cv = GroupKFold(n_splits)
        elif cj.parameters.folds == "stratified":
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        elif cj.parameters.folds != "shuffle":
            raise ValueError("Unknown folding policy '{}'.".format(cj.parameters.folds))

        # Fit
        accuracies = np.zeros(n_splits)
        test_sizes = np.zeros(n_splits)

        _x, _y = pyxit.extract_subwindows(x, y)

        # CV loop
        for i, (train, test) in cj.monitor(enumerate(cv.split(x, y, labels)), start=60, end=90, prefix="cross val. iteration"):
            _pyxit = clone(pyxit)
            w_train = window_indexes(x.shape[0], train, _pyxit.n_subwindows)
            w_test = window_indexes(x.shape[0], test, _pyxit.n_subwindows)
            _pyxit.fit(x[train], y[train], _X=_x[w_train], _y=_y[w_train])
            y_pred = _pyxit.predict(x[test], _x[w_test])
            accuracies[i] = accuracy_score(y[test], y_pred)
            test_sizes[i] = test.shape[0] / float(x.shape[0])
            del _pyxit

        pyxit.fit(x, y)

        accuracy = float(np.mean(test_sizes * accuracies))
        cj.job.update(progress=90, statusComment="Accuracy: {}".format(accuracy))
        cj.job.update(progress=90, statusComment="Save model...")

        model_filename = joblib.dump(pyxit, os.path.join(base_path, "model.joblib"), compress=3)[0]

        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_filename,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        Property(cj.job, key="classes", value=stringify(classes)).save()
        Property(cj.job, key="binary", value=cj.parameters.cytomine_binary).save()
        Property(cj.job, key="positive_classes", value=stringify(positive_classes)).save()
        Property(cj.job, key="accuracies", value=array2str(accuracies))
        Property(cj.job, key="test_sizes", value=array2str(test_sizes))
        Property(cj.job, key="accuracy", value=accuracy)

        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
