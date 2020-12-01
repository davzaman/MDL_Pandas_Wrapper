from typing import List, Dict, Union
from sklearn import datasets
import Orange
import pandas as pd
import numpy as np
from Orange.data import Table, ContinuousVariable, DiscreteVariable


def list2dict(lst: List, names: List[str] = []) -> Dict[Union[str, int], List]:
    if len(names) > 1:
        return {names[i]: lst[i] for i in range(len(lst))}
    else:
        return {i: lst[i] for i in range(len(lst))}


def get_discretized_MDL_data(dataTable: Table, force: bool = True):
    disc = Orange.preprocess.Discretize()
    disc.method = Orange.preprocess.discretize.EntropyMDL(force=force)
    return disc(dataTable), disc


def dfMapColumnValues(df: pd.DataFrame, cols: List[str], dicts: Dict) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].map(dicts[col])
    return df


def orange2Df(
    data: Table, cols: List[str], dicts: Dict, mapped: bool = True
) -> pd.DataFrame:
    X = data.X
    df = pd.DataFrame(data=X, columns=cols)
    return dfMapColumnValues(df, cols, dicts) if mapped else df


def dict2list(d: Dict[str, Dict[int, str]]) -> List[float]:
    """Generates pandas bins from Orange discretizer dicts object."""
    bins = [-np.Inf]
    for i, val in enumerate(d.values()):
        if i == 0:
            bins.append(float(val.replace(" ", "").replace("<", "")))
        elif "-" in val:
            bin1, bin2 = val.replace(" ", "").split("-")
            bins.extend([float(bin1), float(bin2)])
        else:
            bins.append(float(val.replace(" ", "").replace("≥", "")))
    bins.append(np.Inf)
    return np.unique(bins)


def df2Orange(df: pd.DataFrame, y: pd.Series, continuous_cols: List[str]) -> Table:
    class_values = list(y.map(str).unique())
    class_var = DiscreteVariable("class_var", values=class_values)
    domain = Orange.data.Domain(
        [ContinuousVariable(col) for col in continuous_cols], class_vars=class_var
    )

    data = Orange.data.Table.from_numpy(
        domain, pd.concat([df[continuous_cols], y], axis=1).values
    )
    return data


class MDLDiscretizer:
    def __init__(self):
        self.cols = []
        self.disc = None
        self.list_of_values = []
        self.dicts = []

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        continuous_cols: List[str],
        force: bool = True,
    ):
        cont_data = df2Orange(df, y, continuous_cols)
        self._fit(cont_data, force)

    def _fit(self, cont_data: Table, force: bool = True):
        d_cont_data, self.disc = get_discretized_MDL_data(cont_data, force=force)
        self.cols = [attr.name for attr in d_cont_data.domain.attributes]
        self.list_of_values = [attr.values for attr in d_cont_data.domain.attributes]
        self.dicts = list2dict(
            [list2dict(values) for values in self.list_of_values], self.cols
        )

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        continuous_cols: List[str],
        force: bool = True,
        mapped: bool = True,
    ):
        cont_data = df2Orange(df, y, continuous_cols)
        self._fit_transform(cont_data, force, mapped)

    def _fit_transform(self, cont_data: Table, force: bool = True, mapped: bool = True):
        d_cont_data, self.disc = get_discretized_MDL_data(cont_data, force=force)
        self.cols = [attr.name for attr in d_cont_data.domain.attributes]
        self.list_of_values = [attr.values for attr in d_cont_data.domain.attributes]
        self.dicts = list2dict(
            [list2dict(values) for values in self.list_of_values], self.cols
        )
        return orange2Df(
            data=d_cont_data, cols=self.cols, dicts=self.dicts, mapped=mapped
        )

    def transform(
        self, df: pd.DataFrame, class_name: str, mapped: bool = True
    ) -> pd.DataFrame:
        """Transforms df directly, no need to convert to Orange Table.
        Orange has an issue where transform will fit the data again, this is a workaround.
        """
        cont_data = df.copy()  # to keep original df intact

        cols = [col for col in cont_data.columns if col != class_name]

        for col in cols:
            bins = dict2list(self.dicts[col])
            if mapped:
                labels = list(self.dicts[col].values())
            else:
                labels = list(range(len(bins) - 1))
            cont_data.loc[:, col] = pd.cut(
                cont_data[col],
                bins=bins,
                right=True,
                labels=labels,
                include_lowest=True,
            )

        return cont_data


if __name__ == "__main__":
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])

    print("Original data: ")
    print(iris_df[:3])
    print("\n")
    print("Fitting data ...")
    print("\n")
    discritizer = MDLDiscretizer()
    discritizer.fit(iris_df, pd.Series(iris["target"]), iris["feature_names"])
    print("List of discretizations: ")
    print(discritizer.dicts)
    print("\n")
    print("Transformed data: ")
    print(discritizer.transform(iris_df[:3], "target"))
