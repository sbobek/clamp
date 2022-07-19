import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import re
import math
import warnings
import os
import math
import matplotlib.pyplot as plt
import socket
import tempfile

from tqdm import tqdm
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import sklearn.cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, classification_report, roc_auc_score 

from anchor import anchor_tabular
from sklearn.neighbors import NearestNeighbors, KDTree, NearestCentroid
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier


from lux import LUX 
from sklearn.neighbors import NearestNeighbors


class ModelGlobalExplainer():
    def __init__(self,
                 ):
        pass
    
    def fit(self,
            x,
            y,
           point_no_to_global):
        self.explainer = DecisionTreeClassifier(random_state=0,
                                                max_depth = math.floor(max(1, (math.log2(point_no_to_global)))))
        self.explainer.fit(x,y)
        pass
    
    def predict(self,
                X_test, 
                features # keep to be consistent with modellocalexplainer
               ):
        
        self.y_pred = self.explainer.predict(X_test)
        pass

class ModelDTExplainer():
    def __init__(self,
                 ):
        pass
    
    def fit(self,
            x,
            y,
           point_no_to_global):
        
        #print(x)
        self.explainer = DecisionTreeClassifier(random_state=0, max_depth = math.floor(max(1, (math.log2(point_no_to_global)-1))))
        self.explainer.fit(x,y)
        pass
    
    def predict(self,
                X_test, 
                features # keep to be consistent with modellocalexplainer
               ):
        
        self.y_pred = self.explainer.predict(X_test)
        pass

class ModelLocalExplainer():
    def __init__(self, 
        clf: sklearn.base.BaseEstimator,
        class_names: list):
        self.clf = clf
        self.class_names = class_names
        pass
        
    def fit(self, 
            dataset:pd.DataFrame, #dataset that will be used to train the explainer 
            features: list,
            thresh: float
            ):
        
        explainer = self._fit_explainer(dataset.iloc[:,:-1], features)
        self.rules = self._justify_explainer(dataset, explainer, thresh)
        self.hmr_file_name = self.df2hmr(data = self.rules.drop('mult', axis = 1), df_columns_names = features)
        pass

    def predict(self, X_test, features):
        self.y_pred = self._heartdroid(X_test, features)
        pass

    ###HMR file creation
    def inparse(self, condition):
        fs = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\2',condition)
        ss =re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\4',condition)
        res=None
        if fs == '<':
            val = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\1',condition)
            res = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\3 in ['+str(eval(val)+0.001)+r' to \5 ]',condition)
        if ss == '<':
            val = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\5',condition)
            res = re.sub(r'([-+]?[0-9]+\.[0-9]+)(<=|>=|<|>|)(f[0-9]+)(<=|>=|<|>)([-+]?[0-9]+\.[0-9]+)',r'\3 in [\1 to '+str(eval(val)-0.001)+']',condition)
        if res is None:
            return condition
        else:
            return res

    def tohmr(self, series):
        result =[]
        for v in  series.split('AND'):
            v = self.inparse(v.strip().lower().replace(' ',''))
            result.append(v.replace('<=',' lte ')
                          .replace('>=',' gte ').replace('<',' lt ').replace('>',' gt ').replace('=','eq').lower())
        return '['+','.join(result)+']'

    def df2hmr(self, data, df_columns_names):
        numfeats = len(df_columns_names)
        
        types = """xtype [name: float,
    domain: [-10000 to 10000],
    scale: 0,
    base: numeric
    ].
xtype [name: clustertype,
    domain: [0 to 1000],
    scale: 0,
    base: numeric
    ].
"""
        atts_cluster = """
xattr [name: cluster,
    type: clustertype,
    class: simple,
    comm: out
    ].
"""
        atts_placeholder = """
xattr [name: __NAME__,
    type: float,
    class: simple,
    comm: out
    ].
"""
        schema_placeholder = """xschm anchor: [__NAME__] ==> [cluster].
"""
        data['hmr_cond'] = data['Rule'].apply(self.tohmr)  
        data['confidence'] = data['Coverage']*data['Precision']
        atts = ''
        schemacond = []
        for i in df_columns_names:
            atts+=atts_placeholder.replace('__NAME__',i)
            schemacond.append(i)        
        
        schema = schema_placeholder.replace('__NAME__',','.join(schemacond))
        
        f = tempfile.NamedTemporaryFile(mode = 'w', delete=False, suffix='.hmr', dir=os.getcwd())
        f.write(types)
        f.write(atts) 
        f.write(atts_cluster) 
        f.write(schema) 
        for i,r in data.iterrows():
            f.write('xrule anchor/'+str(i)+': '+r['hmr_cond']+ ' ==>  [cluster set '+str(r['Cluster'])+']. #'+str(r['confidence'])+'\n')
        f.close()
        return f.name.split('/')[-1]
        
    def _heartdroid(self, X_test_con, features):
        temp_list = []  
        X_test = X_test_con.values
        model = self.hmr_file_name
        print(model)
        print('Heartdroid run')
        
        for steps in range(X_test.shape[0]):
            finall_string = ''
            for it, index in enumerate(features):
                finall_string += f' -A {index}={X_test[steps][it]}'
                
            output_list = queryHRTDServer(f'{model} -tabs anchor{finall_string}')
            output_list = output_list.split('\n')
            for o in reversed(output_list):
                if 'Attribute cluster' in o:
                    output = [o]
            if 'null' in output[0]:
                temp_list.append(-1) #undefined cluster
            else:
                temp_list.append(int(float(output[0].split(" = ")[-1])))
        temp_list = np.array(temp_list)
        
        if len(temp_list)> 0:
            print("Heardroid success")
            
        return temp_list

class ModelLocalExplainer_anchor(ModelLocalExplainer):    
    
    def _fit_explainer(self, dataset, features):
        predict_fn = lambda x: self.clf.predict_proba(x)
        explainer = AnchorTabular(predict_fn, features)
        explainer.fit(dataset.values, disc_perc=(25, 50, 75))
        return explainer

    def _justify_explainer(self, dataset, explainer, thresh):
        rules_out_list = []
        for cluster in self.class_names:
            rules_out = pd.DataFrame()
            tempo_dataset = dataset[dataset['y'] == cluster]
            for idx in range(tempo_dataset.shape[0]):
                if self.class_names[explainer.predictor(tempo_dataset.iloc[:,:-1].values[idx].reshape(1, -1))[0]] == cluster:
                    explanation = explainer.explain(tempo_dataset.iloc[:,:-1].values[idx], threshold=thresh)
                    exp = explanation.anchor
                    try:
                        rules_out = rules_out.append({'Rule': (' AND '.join(exp)), 
                                                      'Precision': explanation['precision'], 
                                                      'Coverage': explanation['coverage'], 
                                                      'Cluster': cluster, 
                                                      'mult':  explanation['precision']*explanation['coverage']},
                                                      ignore_index = True)
                    except:
                        print('Not possible to create rules for this instance')
            try:        
                df_temporary = rules_out.sort_values('mult', ascending = False).drop_duplicates(subset=['Rule']).reset_index(drop = True)                                                                                           
                rules_out_list.append(df_temporary)
            except:
                print("Nothing to add")
        rules_output = pd.concat(rules_out_list)
        rules_output.reset_index(drop = True, inplace = True)
        return rules_output  
    
class LUXModelLocalExplainer(LUX, ModelLocalExplainer):
    def __init__(self,predict_proba, neighborhood_size=100,max_depth=4,  node_size_limit = 1, grow_confidence_threshold = 0 ):
        self.neighborhood_size=neighborhood_size
        self.max_depth=max_depth
        self.node_size_limit=node_size_limit
        self.grow_confidence_threshold=grow_confidence_threshold
        self.nn = NearestNeighbors(n_neighbors=neighborhood_size)
        self.predict_proba = predict_proba
            
    def fit(self,X,y, instance_to_explain, features, exclude_neighbourhood=False, use_parity=True,class_names=None):
        #features is unused in this explainer, but left in function for consistency
        LUX.fit(self,X,y, instance_to_explain, exclude_neighbourhood=False, use_parity=True,class_names=None)
        list_of_results = [self.to_HMR()]
        self.rules = self._justify_exp(list_of_results)
        self.hmr_file_name = self.df2hmr(data = self.rules.drop('mult', axis = 1), df_columns_names = features)
   
    def predict(self,X,y=None):
        self.y_pred = [int(p) for p in LUX.predict(self, X,y)]
        
    def hmr_final(self, list_of_res, columns):
        
        rules = []
        for res in list_of_res:
            rules.append(re.findall('\[.+#.*',res))       
    
        types = """xtype [name: float,
                    domain: [-10000 to 10000],
                    scale: 0,
                    base: numeric
                    ].
                xtype [name: clustertype,
                    domain: [0 to 1000],
                    scale: 0,
                    base: numeric
                    ].
                """
        atts_cluster = """
                xattr [name: cluster,
                    type: clustertype,
                    class: simple,
                    comm: out
                    ].
                """
        atts_placeholder = """
                xattr [name: __NAME__,
                    type: float,
                    class: simple,
                    comm: out
                    ].
                """
        schema_placeholder = """xschm anchor: [__NAME__] ==> [cluster].\n"""
        atts = ''
        schemacond = []
        for i in columns:
            atts+=atts_placeholder.replace('__NAME__',i)
            schemacond.append(i)        
                
        schema = schema_placeholder.replace('__NAME__',','.join(schemacond))
        f = tempfile.NamedTemporaryFile(mode = 'w', delete=False, suffix='.hmr', dir=os.getcwd())
        f.write(types)
        f.write(atts) 
        f.write(atts_cluster) 
        f.write(schema) 
        
        rules_flatten = sum(rules, [])
        for i,rule in enumerate(rules_flatten):
            f.write('xrule anchor/'+str(i)+': '+str(rule)+'\n')
        f.close()
        
        return f.name.split('/')[-1]
    
    def _justify_exp(self, list_of_res):
        
        rules = []
        for res in list_of_res:
            rules.append(re.findall('\[.+#.*',res)) 
        
        rules_flatten = sum(rules, [])
        rules_out = pd.DataFrame()
        for i,rule in enumerate(rules_flatten):
            rules_out = rules_out.append(
            {'Rule': rule.replace(' lte ','<= ').replace(' gte ','>= ').replace(' lt ','< ').replace(' gt ','> ').replace('eq','= ').replace(',',' AND ').split(']')[0][1:],
             'Precision': 1,
             'Coverage' :rule.replace(' lte ','<= ').replace(' gte ','>= ').replace(' lt ','< ').replace(' gt ','> ').replace('eq','= ').replace(',',' AND ').split('#')[-1],
             'Cluster': rule.replace(' lte ','<= ').replace(' gte ','>= ').replace(' lt ','< ').replace(' gt ','> ').replace('eq','= ').replace(',',' AND ').split(']')[1][-1],
             'mult':rule.replace(' lte ','<= ').replace(' gte ','>= ').replace(' lt ','< ').replace(' gt ','> ').replace('eq','= ').replace(',',' AND ').split('#')[-1]}, ignore_index = True)
        rules_out.sort_values(by = 'Cluster', inplace = True)
        rules_out.reset_index(drop = True, inplace = True)
        df_temporary = rules_out.sort_values('mult', ascending = False).drop_duplicates(subset=['Rule']).reset_index(drop = True)                                                                                           
        return df_temporary
    
    

    @staticmethod
    def generate_uarff(X,y,class_names):
        uarff="@relation lux\n\n"
        for f,t in zip(X.columns,X.dtypes):
            if t in (int, float):
                uarff+=f'@attribute {f} @REAL\n'
            else:
                domain = ','.join(list(X[f].nunique()))
                uarff+='@attribute '+f+'{'+domain+'}\n'
        

        domain = ','.join([str(cn) for cn in class_names])
        uarff+='@attribute class {'+domain+'}\n'
        
        uarff += '@data\n'
        for i in range(0, X.shape[0]):
            for j in range(0,X.shape[1]):
                uarff+='{:.2f}'.format(X.iloc[i,j])+'[1],'
            uarff+=';'.join([f'{c}[{p}]' for c,p in zip(class_names, y[i,:])])+'\n'
        return uarff
   
     
class CLAMP(BaseEstimator):
    def __init__(self,
        bounding_box_selection: str='random',
        classification_model: sklearn.base.BaseEstimator = LogisticRegression(), 
        clusterng_algorithm: sklearn.base.BaseEstimator = KMeans(),
        description_points_ratio: float=5,
        test_size: float=0.2,
        metric: str='minkowski',
        explainer_type: str='anchor',
        thresh: float=0.9,
        conv_method: str = None,
        approach: str = 'obo',
        neighborhood_size: int = 3,
        max_depth: int=4,  
        node_size_limit: int = 1, 
        grow_confidence_threshold: int = 0
        ):
        
        self.bounding_box_selection = bounding_box_selection
        self.classification_model = classification_model
        self.clusterng_algorithm = clusterng_algorithm
        self.description_points_ratio = description_points_ratio
        self.test_size = test_size
        self.metric = metric
        self.explainer_type=explainer_type
        self.thresh = thresh
        self.hrd_accuracy = 0
        self.conv_method = None
        self.approach = approach
        self.neighborhood_size = neighborhood_size
        self.max_depth = max_depth
        self.node_size_limit = node_size_limit
        self.grow_confidence_threshold = grow_confidence_threshold
        
        pass
         
    def fit(self,
        x_in : pd.DataFrame, #data which will be used to train explainer model and classifier
        y: pd.Series=None, # cluster labels (not used, left for consistency with BaseEstimator)
        ):
        """
        #fits the Clustering algorithm and classifier
        """
        
        #exchange column names
        x, self.orignal_features = self._convert_features(x_in)
        if y is None:
            #clustering stage
            y = self._clustering(x) # only if y not in data
            print('Data without labels, clustering stage implementation')
        else:
            y = np.array(y)
            print('Data labeled')

        #classification stage
        self.X_train, self.X_test, self.y_train, self.y_test = self._recognize_input(x, y)
        
        self.point_no_to_global = math.floor(self.description_points_ratio)
        self.X_train = self._convert_to_norm(self.X_train, self.conv_method)
        self.X_test = self._convert_to_norm(self.X_test, self.conv_method)
        
        self.y_pred_clf, self.clf_model = self._classification()
        self.clf_precision, self.clf_recall, self.clf_f1, self.clf_accuracy, self.clf_classification_report = self._scores(self.y_test, self.y_pred_clf)        
        
        if self.approach == 'obo':
            self.df_model_input =  self._bounding_box_method(self.X_train, self.y_train)

            if self.explainer_type == 'anchor':
                self.explainer = ModelLocalExplainer_anchor(clf = self.clf_model, class_names = self.class_names)
                print('Anchor explainer')
                self.explainer.fit(dataset=self.df_model_input, features = self.features, thresh = self.thresh)

            elif self.explainer_type == 'global':
                self.explainer = ModelGlobalExplainer()
                print('DT global')
                self.explainer.fit(x=self.X_train, y=self.y_train, point_no_to_global = self.point_no_to_global)

            elif self.explainer_type == 'dtexp':
                self.explainer = ModelDTExplainer()
                print('DT explainer')
                self.explainer.fit(x=self.df_model_input.iloc[:,:-1], y=self.df_model_input.iloc[:,-1], point_no_to_global = self.point_no_to_global)
            
            elif self.explainer_type == 'lux':
                self.explainer = LUXModelLocalExplainer(predict_proba = self.clf_model.predict_proba, neighborhood_size=self.neighborhood_size,max_depth=self.max_depth,  node_size_limit = self.node_size_limit, grow_confidence_threshold = self.grow_confidence_threshold)
                print('Lux explainer')
                self.explainer.fit(self.X_train, self.y_train, self.df_model_input.iloc[:,:-1].to_numpy(), features = self.features, class_names=self.class_names)
                
            else:
                ValueError('Explainer type not implemented. Select one of: anchor, global.')
            pass
            
        else:
            print("Brute approach, choosen data description skipped. All cases will be checked.")
            dataset_descroption_method = ['random', 'centroids', 'outliers', 'tree_query']
            all_scores = pd.DataFrame()
            rules_dict = dict()
            final_results_rules = pd.DataFrame()
            for description_method in dataset_descroption_method:
                print('Method: ', description_method)
                self.bounding_box_selection = description_method
                self.df_model_input = self._bounding_box_method(self.X_train, self.y_train)
                
                if self.explainer_type == 'anchor':
                    self.explainer = ModelLocalExplainer_anchor(clf = self.clf_model, class_names = self.class_names)
                    print('Anchor explainer')
                    self.explainer.fit(dataset=self.df_model_input, features = self.features, thresh = self.thresh)
                    self.explainer.predict(self.X_test, self.features)
                    
                if self.explainer_type == 'lux':  
                    self.explainer = LUXModelLocalExplainer(predict_proba = self.clf_model.predict_proba, neighborhood_size=self.neighborhood_size,max_depth=self.max_depth,  node_size_limit = self.node_size_limit, grow_confidence_threshold = self.grow_confidence_threshold)
                    print('Lux explainer')
                    self.explainer.fit(self.X_train, self.y_train, self.df_model_input.iloc[:,:-1].to_numpy(), features = self.features, class_names=self.class_names)
                    self.explainer.predict(self.X_test) # self.y_test
                    
                self.y_pred_explainer = self.explainer.y_pred  
                raport = pd.DataFrame(classification_report(self.y_test, self.y_pred_explainer, output_dict=True)).transpose().iloc[:-3,:-1]['f1-score']
                print('Accuracy:', accuracy_score(self.y_test, self.y_pred_explainer))
                raport.name = description_method
                all_scores = pd.concat([all_scores, raport[raport.index != '-1']], axis = 1)
                rules_dict[description_method] = self.explainer.rules
            best_description_methods = all_scores.idxmax(axis=1).to_dict()
            
            for cluster, method in best_description_methods.items():
                final_results_rules = pd.concat([final_results_rules, rules_dict[method][rules_dict[method]['Cluster'] == cluster][['Rule', 'Cluster','Precision','Coverage', 'mult']]])
            final_results_rules.reset_index(drop = True, inplace = True)
            
            self.explainer.hmr_file_name = self.explainer.df2hmr(data = final_results_rules, df_columns_names = self.features)
            

    def justify(self):#, X_test, y_test):
        return self._convert_rules(self.explainer.rules.drop('mult', axis = 1), self.orignal_features, self.features)

    def predict(self, X_test, y_test):
        if self.explainer_type == 'lux':  
            self.explainer.predict(self.X_test) # self.y_test
        else: 
            self.explainer.predict(X_test, self.features)
        self.y_pred_explainer = self.explainer.y_pred
        self.explainer_precision, self.explainer_recall, self.explainer_f1, self.explainer_accuracy, self.explainer_classification_report = self._scores(y_test, self.y_pred_explainer)
        print('Accuracy:', accuracy_score(y_test, self.y_pred_explainer))

        pass
    
    def _convert_rules (self, rules, org, mod):
        my_dict = {} 
        for key in mod: 
            for value in org: 
                my_dict[key] = value 
                org.remove(value) 
                break 
        rules_con_org_features = []
        for rule in rules['Rule']:
            temp = rule.split()
            res = []
            for wrd in temp:
                res.append(my_dict.get(wrd, wrd))
            rules_con_org_features.append(' '.join(res))
        rules['Rule'] = rules_con_org_features
        return rules
    
    def _convert_features(self, data):
        orignal_features = list(data.columns)
        data.columns = ['f' + str(list(data.columns).index(x)) for x in list(data.columns)]
        return data, orignal_features
    
    def _clustering(self, x):       
        try:
            clustering_model = self.clusterng_algorithm.fit(x)
            prediction = clustering_model.predict(x)
        except:
            prediction = self.clusterng_algorithm.fit_predict(x)
        return prediction
    
    def _recognize_input(self, x, y):
        self.features = x.columns
        self.num_of_features = len(self.features)
        self.class_names = np.unique(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=0)
        return X_train, X_test, y_train, y_test
    
    
    ####not used in grid search
    def _convert_to_norm(self,x, method):
        if method == 'standard_scaler':
            scaler = StandardScaler()
            scaler.fit(x)
            converted_data = scaler.transform(x)
        elif method == 'minmax_scaler':
            scaler = MinMaxScaler()
            scaler.fit(x)
            converted_data = scaler.transform(x)
        elif method == None:
            converted_data = x
        return converted_data
    
    def _classification(self):   
        classification_model = self.classification_model.fit(self.X_train, self.y_train)
        return classification_model.predict(self.X_test), classification_model
    
    def _scores(self, y_test, y_pred):
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, labels=self.class_names)
        return precision, recall, f1, accuracy, classification_rep#, auc
    
    def _bounding_box_method(self, x, y):
        data = pd.concat([x.reset_index(drop = True), pd.Series(y)], axis=1)
        data.columns = list(self.features)+['y']
        number_of_points = math.floor(self.description_points_ratio)
        if number_of_points > data.shape[0]:
            number_of_points = data.shape[0]//len(self.class_names)
        self.point_no_to_global = number_of_points
        temp_list = []
        if self.bounding_box_selection == 'random':
            for cluster in self.class_names:
                X_t = data[data['y'] == cluster]
                try:
                    temp_list.append(X_t.sample(n = number_of_points))
                except:
                    temp_list.append(X_t.sample(n = X_t.shape[0]))
                    
        elif self.bounding_box_selection == 'tree_query':
            clf = NearestCentroid()
            clf.fit(data.drop('y', axis = 1).to_numpy(), data['y'].to_numpy())
            
            df_centroids = pd.DataFrame(clf.centroids_, columns = self.features, index = clf.classes_)
            for cluster in self.class_names:
                X_t = data[data['y'] == cluster].drop('y', axis = 1).values
                tree = KDTree(X_t, leaf_size = 10, metric = self.metric)  
                dist, ind = tree.query(df_centroids.loc[cluster].values.reshape(1,-1), k=len(X_t))      
                temp_df = data[data['y'] == cluster].iloc[ind[0][-number_of_points:],:]
                temp_list.append(temp_df)
        
        elif self.bounding_box_selection == 'outliers':
            for cluster in self.class_names:
                cluster_cont = number_of_points/(data[data['y'] == cluster].shape[0])
                if cluster_cont > 0.5:
                    print("number of description points cannot be higher than 50%, value has been changed to maximum")
                    cluster_cont = 0.5
                random_data = data[data['y'] == cluster].drop('y', axis = 1).values
                clf = IsolationForest(random_state = 0, contamination= cluster_cont)
                preds = clf.fit_predict(random_data)
                df_temp = pd.DataFrame(random_data[[i for i, x in enumerate(preds) if x == -1]], columns = self.features)
                df_temp['y'] = [cluster] * len(df_temp)
                temp_list.append(df_temp)
                
                
        elif self.bounding_box_selection == 'centroids':
            for cluster in self.class_names:
                X_t = data[data['y'] == cluster].drop('y', axis = 1)
                X_t.dropna(inplace = True)
                X_t = X_t.to_numpy()
                kmedoids = KMedoids(n_clusters=1, random_state=0).fit(X_t)
                
                df_temp = pd.DataFrame(data = kmedoids.cluster_centers_, columns = data.drop('y', axis = 1).columns)
                df_temp['y'] = [cluster] * len(df_temp)
                temp_list.append(df_temp)
        else:
            ValueError('Bounding box method not implemented. Select one of: random, tree_query, outliers.')
        df_ready = pd.concat(temp_list)
        df_ready.reset_index(inplace = True, drop = True)
        
        return df_ready
 
    def _create_necessary_dir(self):
        path = pathlib.Path().resolve() / 'hmr_models'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)