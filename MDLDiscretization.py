import Orange
import pandas as pd
import numpy as np
import csv
from io import StringIO
from collections import OrderedDict
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable


########################################### MDL discretization


def list2dict(lst,names = []):
    if len(names)>1:
        return {names[i]: lst[i] for i in range(len(lst))}
    else:
        return {i: lst[i] for i in range(len(lst))}

def get_descritized_MDL_data(dataTable,force=True):
    disc = Orange.preprocess.Discretize()
    disc.method = Orange.preprocess.discretize.EntropyMDL(force=force)
    return disc(dataTable),disc

def dfMapColumnValues(df,cols,dicts):
    for col in cols:
        df[col] = df[col].map(dicts[col])
    return df

def orange2Df(data, cols, dicts, mapped=True):
    X = data.X

    df = pd.DataFrame(data=X,columns=cols)
    
    if mapped:
        return dfMapColumnValues(df,cols,dicts)
    else:
        return df



def df2Orange(df,class_name):
    
    #class values to string
    df[class_name] = [str(i) for i in df[class_name].tolist()]
    class_values = list(df[class_name].unique())

    cols = [col for col in df.columns if col!=class_name]
    
    class_var = DiscreteVariable("class_var", values=class_values)
    domain = Orange.data.Domain([ContinuousVariable(col) for col in cols],
                                class_vars=class_var)

    data = Orange.data.Table(domain,df.values.tolist())
    
    return data

def dict2list(d):
    bins = [-np.Inf]
    for i,val in enumerate(d.values()):
        if i == 0:
            bins.append(float(val.replace(' ','').replace('<','')))
        elif '-' in val:
            bin1, bin2 = val.replace(' ','').split('-')
            bins.extend([float(bin1),float(bin2)])
        else:
            bins.append(float(val.replace(' ','').replace('≥','')))
    bins.append(np.Inf)
    return np.unique(bins)


class MDLDescritizer:
    def __init__(self):
        self.cols = []
        self.disc = None
        self.list_of_values = []
        self.dicts = []

    def fit(self, cont_data, force=True):
        d_cont_data,self.disc = get_descritized_MDL_data(cont_data, force=force)
        self.cols = [attr.name for attr in d_cont_data.domain.attributes]
        self.list_of_values = [attr.values for attr in d_cont_data.domain.attributes]
        self.dicts = list2dict([list2dict(values) for values in self.list_of_values]
                                ,self.cols)
        return;

    def fit_transform(self, cont_data, force=True, mapped=True):
        d_cont_data,self.disc = get_descritized_MDL_data(cont_data, force=force)
        self.cols = [attr.name for attr in d_cont_data.domain.attributes]
        self.list_of_values = [attr.values for attr in d_cont_data.domain.attributes]
        self.dicts = list2dict([list2dict(values) for values in self.list_of_values]
                                ,self.cols)

        return orange2Df(data=d_cont_data, cols=self.cols, dicts=self.dicts, mapped=mapped)
    

    def transform(self, df, mapped=True, class_name='target'):
       
        cont_data = df.copy() #to keep original df intact 
        
        cols = [col for col in cont_data.columns if col != class_name]
        
        for col in cols:
            bins = dict2list(self.dicts[col])
            if mapped:
                labels = list(self.dicts[col].values())
            else:
                labels = list(range(len(bins)-1))
            cont_data.loc[:,col] = pd.cut(cont_data[col], bins=bins, right=True, 
                                    labels=labels, include_lowest=True)
            
        return cont_data    


from sklearn import datasets


if __name__ == "__main__":
    iris = datasets.load_iris()
    iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

    iris_orange = df2Orange(iris,class_name='target')

    print('Original data: ')
    print(iris_orange[:3])
    print('\n')
    print('Fitting data ...')
    print('\n')
    discritizer = MDLDescritizer()
    discritizer.fit(cont_data=iris_orange)
    print('List of discretizations: ')
    print(discritizer.dicts)
    print('\n')
    print('Transformed data: ')
    print(discritizer.transform(cont_data=iris)[:3])


  


