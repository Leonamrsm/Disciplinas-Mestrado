from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, auc, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.validation import check_X_y
import random
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston

from mlxtend.classifier import Adaline

from sklearn.svm import SVR

class RBF(BaseEstimator, RegressorMixin):
    """ Classificador RBF """

    def __init__(self, p = 2 , lambdar = 0, classificator = True):
        self.W = None
        self.p = p
        self.m = None
        self.covlist = []
        self.n = None
        self.lambdar = lambdar
        self.classificator = classificator


    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""

        check_X_y(X, y)

        if self.classificator:
            assert set(y) == {-1, 1}, 'Response variable must be ±1'

        return X, y

    def pdfnvar(self, x, m, K):
        """Função radial Gaussiana
            Parâmetros
            ----------
            x: amostra de forma (1, n_características)
            m: vetor de médias de forma (n_características,)
            K: matriz de covariâncias de forma (n_características, n_características)
            Retorna
            -------
            p: pdf para cada entrada em um dado cluster determinado po m e K
        """
        p = 1/np.sqrt((2*np.pi) ** self.n * np.linalg.det(K)) * np.exp((-0.5*(x - m).T).dot(np.linalg.pinv(K)).dot(x-m))

        return p

    def calcH(self, X: np.ndarray):
        """Função que cálcula a matriz H a a partir dos valores de centros
        e as matrizes de covariâncias de cada centro do modelo
            Parâmetros
            ----------
            X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
                Os exemplos de entrada de treinamento.
            Retorna
            -------
            H: matriz H (saída da pdf de cada neurônio para cada amostra acrescida de um bias)
        """

        # número de amostras
        N = X.shape[0]

        H = np.ones((N, self.p + 1))

        for j in range(N):
            for i in range(self.p):
                mi = self.m[i,:]
                covi = self.covlist[i] + 0.001 * np.identity(X.shape[1])
                H[j,i +1] = self.pdfnvar(X[j,:], mi, covi)

        return H

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Controi um classificador otimizado a partir do conjunto de treinamento (X, y).
            Parâmetros
            ----------
            X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
                Os exemplos de entrada de treinamento.
            y: Vetor de formato (n_samples,)
                Os valores alvo (rótulos de classe) do conjunto de treinamento.

            p: número de neurônios da camada intermediária
            lambdar:  termo de penalização
            Retorna
            -------
            self: objeto
                Estimador ajustado.
            """

        # Valida os rótulos
        X, y = self._check_X_y(X, y)

        # dimensão de entrada
        self.n = X.shape[1]

        # Calcula K-médias para a entrada X
        xclust = KMeans(n_clusters=self.p).fit(X)

        # Armazena vetores de centros das funções.
        self.m = xclust.cluster_centers_

        # Estima matrizes de covariância para todos os centros.
        for i in range(self.p):
            xci = X[(xclust.labels_== i),:]
            covi = np.cov(xci, rowvar=False)
            self.covlist.append(covi)

        # Calcula matriz H
        H = self.calcH(X)

        # Calcula matriz de variância A
        A = np.dot(H.T,H) + self.lambdar*np.identity(self.p+1)

        # Vetor pesos entre a camada de saída e a camada intermediária
        self.W = np.dot(np.dot(np.linalg.inv(A), H.T), y)

        return self

    def predict(self, X):
        """ Faz previsões usando o modelo já ajustado
        Parâmetros
            ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
            Os exemplos de entrada.
        """

        # Calcula matriz H
        H = self.calcH(X)
        if self.classificator:
            return np.sign(np.dot(H,self.W))
        else:
            return np.dot(H,self.W)


    def score(self, X, Y):
        """ Retorna a acurácia do classificador ou r^2 score da regressão
        Parâmetros
            ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
            Os exemplos de entrada.
        Y: Vetor de formato (n_samples,)
            Os valores alvo (rótulos de classe) dos exemplos de entrada.
        """
        Y_pred = self.predict(X)
        if self.classificator:
            return np.sum(Y_pred == Y)/Y.shape[0]
        else:
            return super().score(X, Y)
            # return - self.costFunction(X, Y)

    def costFunction(self, X, Y):
        """ Retorna o custo de aproximação do calssificador, contendo o custo devido a
            diferença média quadrática e o custo relativo à penalização dos pesos
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        Y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) dos exemplos de entrada.
        Retorna
        -------
        Jc: Erro devido à função de custos
        Jw: Erro devido à penalização dos pesos
        J: Erro total
        """

        H = self.calcH(X)

        # Calcula matriz de variância A
        A = np.dot(H.T,H) + self.lambdar * np.identity(self.p +1)

        # Calcula matriz de projeção P
        P = np.identity(X.shape[0]) - np.dot(np.dot(H, np.linalg.inv(A)), H.T)

        # Custo devido à diferença média quadrática
        # Jc = np.dot(np.dot(Y.T, np.dot(P, P)), Y)

        # Custo devido à penalização dos pesos
        # Jw = np.dot(np.dot(Y.T, P - np.dot(P, P)), Y)

        # Calcula o custo total
        J = np.dot(np.dot(Y.T, P), Y)

        # Retorna o custo médio
        return J/X.shape[0]

class ELM(BaseEstimator, RegressorMixin):
    """ Classificador ELM """

    def __init__(self, p = 2, lambdar = 0, classificator  = True):
        self.Z = None
        self.W = None
        self.p = p
        self.lambdar = lambdar
        self.classificator = classificator


    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""

        check_X_y(X, y)

        if self.classificator:
            assert set(y) == {-1, 1}, 'Response variable must be ±1'

        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Controi um classificador otimizado a partir do conjunto de treinamento (X, y).
         Parâmetros
         ----------
         X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada de treinamento.
         y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) do conjunto de treinamento.

         p: número de neurônios da camada intermediária
         lambdar:  termo de penalização
         Retorna
         -------
         self: objeto
             Estimador ajustado.
         """
        # Valida os rótulos
        X, y = self._check_X_y(X, y)

        # Vetor pesos da camada escondida gerado de forma aleatória
        self.Z = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[1] +1, self.p))

        # Calcula matriz H
        H = np.tanh(np.dot(X, self.Z[1:]) + self.Z[0])

        # Calcula matriz de variância A
        A = np.dot(H.T,H) + self.lambdar*np.identity(self.p)

        # Vetor pesos entre a camada de saída e a camada intermediária
        self.W = np.dot(np.dot(np.linalg.inv(A), H.T), y)

        return self

    def predict(self, X):
        """ Make predictions using already fitted model
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        """

        H = np.tanh(np.dot(X, self.Z[1:]) + self.Z[0])
        if self.classificator:
            return np.sign(np.dot(H,self.W))
        else:
            return np.dot(H,self.W)


    def score(self, X, Y):
        """ Retorna a acurácia do classificador
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        Y: Vetor de formato (n_amostras,)
             Os valores alvo (rótulos de classe) dos exemplos de entrada.
        """
        Y_pred = self.predict(X)
        if self.classificator:
            return np.sum(Y_pred == Y)/Y.shape[0]
        else:
            return super().score(X, Y)
            # return - self.costFunction(X, Y)

    def costFunction(self, X, Y):
        """ Retorna o custo de aproximação do calssificador, contendo o custo devido a
            diferença média quadrática e o custo relativo à penalização dos pesos
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        Y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) dos exemplos de entrada.
        Retorna
        -------
        Jc: Erro devido à função de custos
        Jw: Erro devido à penalização dos pesos
        J: Erro total
        """

        H = np.tanh(np.dot(X, self.Z[1:]) + self.Z[0])

        # Calcula matriz de variância A
        A = np.dot(H.T,H) + self.lambdar * np.identity(self.p)

        # Calcula matriz de projeção P
        P = np.identity(X.shape[0]) - np.dot(np.dot(H, np.linalg.inv(A)), H.T)

        # Custo devido à diferença média quadrática
        # Jc = np.dot(np.dot(Y.T, np.dot(P, P)), Y)

        # Custo devido à penalização dos pesos
        # Jw = np.dot(np.dot(Y.T, P - np.dot(P, P)), Y)

        # Calcula o custo total
        J = np.dot(np.dot(Y.T, P), Y)

        # Retorna o custo médio
        return J/X.shape[0]


class ELMHebbiano(BaseEstimator, RegressorMixin):
    """ Classificador ELM Hebbiano"""

    def __init__(self, p = 2, classificator = True):
        self.Z = None
        self.W = None
        self.p = p
        self.classificator = classificator


    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""

        check_X_y(X, y)

        if self.classificator:
            assert set(y) == {-1, 1}, 'Response variable must be ±1'

        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Controi um classificador otimizado a partir do conjunto de treinamento (X, y).
         Parâmetros
         ----------
         X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada de treinamento.
         y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) do conjunto de treinamento.

         p: número de neurônios da camada intermediária
         lambdar:  termo de penalização
         Retorna
         -------
         self: objeto
             Estimador ajustado.
         """
        # Valida os rótulos
        X, y = self._check_X_y(X, y)

        # Vetor pesos da camada escondida gerado de forma aleatória
        self.Z = np.random.uniform(low=-0.5, high=0.5, size=(X.shape[1] +1, self.p))

        # Calcula matriz H
        H = np.tanh(np.dot(X, self.Z[1:]) + self.Z[0])

        # Vetor pesos entre a camada de saída e a camada intermediária
        self.W = np.dot(H.T, y)/np.linalg.norm(np.dot(H.T, y))

        return self

    def predict(self, X):
        """ Make predictions using already fitted model
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        """

        H = np.tanh(np.dot(X, self.Z[1:]) + self.Z[0])

        if self.classificator:
            return np.sign(np.dot(H,self.W))
        else:
            return np.dot(H,self.W)


    def score(self, X, Y):
        """ Retorna a acurácia do classificador
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        Y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) dos exemplos de entrada.
        """
        Y_pred = self.predict(X)

        if self.classificator:
            return np.sum(Y_pred == Y)/Y.shape[0]
        else:
            return super().score(X, Y)

    def costFunction(self, X, Y):
        """ Retorna o custo de aproximação do calssificador, contendo o custo devido a
            diferença média quadrática e o custo relativo à penalização dos pesos
        Parâmetros
         ----------
        X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
             Os exemplos de entrada.
        Y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) dos exemplos de entrada.
        Retorna
        -------
        Jc: Erro devido à função de custos
        Jw: Erro devido à penalização dos pesos
        J: Erro total
        """

        H = np.tanh(np.dot(X, self.Z[1:]) + self.Z[0])

        # Calcula matriz de variância A
        A = np.dot(H.T,H)

        # Calcula matriz de projeção P
        P = np.identity(X.shape[0]) - np.dot(np.dot(H, np.linalg.inv(A)), H.T)

        # Custo devido à diferença média quadrática
        # Jc = np.dot(np.dot(Y.T, np.dot(P, P)), Y)

        # Custo devido à penalização dos pesos
        # Jw = np.dot(np.dot(Y.T, P - np.dot(P, P)), Y)

        # Calcula o custo total
        J = np.dot(np.dot(Y.T, P), Y)

        # Retorna o custo médio
        return J/X.shape[0]


# class Adaline(RegressorMixin):
#     """ Classificador Adaline """

#     def __init__(self, eta = 0.00001, tol = 0.01, maxepocas = 1000, lambdar = 0, k = 50):
#         self.W = None
#         self.lambdar = lambdar
#         self.eta = eta
#         self.tol = tol
#         self.maxepocas = maxepocas
#         self.k = k


#     def _check_X_y(self, X, y):
#         """ Validate assumptions about format of input data"""

#         check_X_y(X, y)

#         return X, y

#     def fit(self, X: np.ndarray, y: np.ndarray):
#         """Controi um classificador otimizado a partir do conjunto de treinamento (X, y).
#          Parâmetros
#          ----------
#          X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
#              Os exemplos de entrada de treinamento.
#          y: Vetor de formato (n_samples,)
#              Os valores alvo (rótulos de classe) do conjunto de treinamento.

#          p: número de neurônios da camada intermediária
#          lambdar:  termo de penalização
#          Retorna
#          -------
#          self: objeto
#              Estimador ajustado.
#          """
#         # Valida os rótulos
#         X, y = self._check_X_y(X, y)


#         dimxi = X.shape
#         N = dimxi[0]
#         n = dimxi[1]

#         self.W = np.random.uniform(low=-0.5, high=0.5, size=n+1)
#         X = np.hstack((np.ones((N ,1)), X))

#         nepoca = 0
#         eepoca = self.tol + 1

#         while((nepoca < self.maxepocas) and (eepoca > self.tol)):
#             # y_pred = np.dot(X, self.W)
#             # e =  y_pred - y 
#             # dw = (1/N) * (np.dot(X.T, e) + self.lambdar * self.W)
#             # self.W = self.W - self.eta * dw
#             # nepoca += 1
#             # ei2 = 0
#             xseq = random.sample(range(N), self.k)

#             w_gradient = np.zeros((n+1,))

#             for i in range(self.k): # Small Batch of size K
#                 irand = xseq[i]
#                 prediction = np.dot(X[irand,:], self.W)
#                 ei = y[irand] - prediction
#                 dw = -2 * ei * X[irand,:] + self.lambdar * self.W
#                 w_gradient = w_gradient + dw

#             self.W = self.W + self.eta * (w_gradient/self.k)

#             # for i in range(N): # Calculating gradients for point in our K sized dataset
                
#             #     irand = xseq[i]

#             #     prediction = np.dot(X[irand,:], self.W)

#             #     ei = y[irand] - prediction

#             #     dw = 

#             #     self.W = self.W + self.eta * dw

#             #     ei2 = ei2 + ei * ei

#             self.eta/=1.2
#             nepoca += 1

#         return self



#     def predict(self, X):
#         """ Make predictions using already fitted model
#         Parâmetros
#          ----------
#         X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
#              Os exemplos de entrada.
#         """
#         X = np.hstack((np.ones((X.shape[0] ,1)), X))

#         return np.dot(X, self.W)


#     def score(self, X, Y):
#         """ Retorna a acurácia do classificador
#         Parâmetros
#          ----------
#         X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
#              Os exemplos de entrada.
#         Y: Vetor de formato (n_samples,)
#              Os valores alvo (rótulos de classe) dos exemplos de entrada.
#         """

#         return super().score(X, Y)

def run_cross_validation_on_elms(X, y, Nneuronios, lambdas, model, loo = False, cv=10):
    """ Método para ajustar ELMs com diferentes números de neurônios na camada oculta
        sobre os dados de treinamento usando validação cruzada

    Parâmetros
        ----------
    X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
            Os exemplos de entrada.

    y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) do conjunto de treinamento.

    Nneuronios: Lista com quantidade de neurônios (p) que serão avaliados

    lambdas: Lista com quantidade de fatores de regularização que serão avaliados



    model: classe do modelo que será utilizado (ELM, RBF ou ELM Hebbiano)

    loo: Caso True será calculado o leaving one out cross validation

    cv: Número de folds que serão utilizados na validação cruzada, caso loo = False

    """

    if loo:
        cv = X.shape[0]

    # Listas auxiliares que serão retornadas
    cv_scores_std = np.zeros((len(Nneuronios), len(lambdas)))
    cv_scores_mean = np.zeros((len(Nneuronios), len(lambdas)))


    for i, n in enumerate(Nneuronios):
        for j, l in enumerate(lambdas):
            ELM_model = model(n, l)
            cv_scores = cross_val_score(ELM_model, X, y, cv=cv, scoring='accuracy')
            cv_scores_mean[i][j] = cv_scores.mean()
            cv_scores_std[i][j] = cv_scores.std()
    return cv_scores_mean, cv_scores_std


def run_cross_validation_on_perceptron(X, y, lambdas, loo = False, cv=10):
    """ Método para ajustar ELMs com diferentes números de neurônios na camada oculta
        sobre os dados de treinamento usando validação cruzada

    Parâmetros
        ----------
    X: {tipo matriz, matriz esparsa} de forma (n_amostras, n_características)
            Os exemplos de entrada.

    y: Vetor de formato (n_samples,)
             Os valores alvo (rótulos de classe) do conjunto de treinamento.

    Nneuronios: Lista com quantidade de neurônios (p) que serão avaliados

    lambdas: Lista com quantidade de fatores de regularização que serão avaliados



    model: classe do modelo que será utilizado (ELM, RBF ou ELM Hebbiano)

    loo: Caso True será calculado o leaving one out cross validation

    cv: Número de folds que serão utilizados na validação cruzada, caso loo = False

    """

    if loo:
        cv = X.shape[0]

    # Listas auxiliares que serão retornadas
    cv_scores_std = np.zeros((len(lambdas),))
    cv_scores_mean = np.zeros((len(lambdas),))


    for i, l in enumerate(lambdas):
        model = Perceptron(penalty='l2', alpha=l)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_scores_mean[i] = cv_scores.mean()
        cv_scores_std[i] = cv_scores.std()
    return cv_scores_mean, cv_scores_std


def getMeanAUC(model, X_train, Y_train, X_test, Y_test, NNeuronios, lambdar, iterations = 10):

    aucs = np.zeros(iterations)

    for i in range(iterations):
        classifier_model = model(NNeuronios, lambdar)
        classifier_model.fit(X_train, Y_train)
        aucs[i] = roc_auc_score(Y_test, classifier_model.predict(X_test))

    return np.mean(aucs), np.std(aucs)

def getMeanAUC_Perceptron(X_train, Y_train, X_test, Y_test, lambdar, iterations = 10):

    aucs_array = np.zeros(iterations)

    for i in range(iterations):
        classifier_model = Perceptron(penalty='l2', alpha=lambdar)
        classifier_model.fit(X_train, Y_train)
        aucs_array[i] = roc_auc_score(Y_test, classifier_model.predict(X_test))

    return np.mean(aucs_array), np.std(aucs_array)


def getRegressionScores(model, X_train, Y_train, X_test, Y_test, NNeuronios, lambdar, iterations = 10):

    r2s = np.zeros(iterations)
    costs = np.zeros(iterations)

    for i in range(iterations):
        classifier_model = model(NNeuronios, lambdar, classificator=False)
        classifier_model.fit(X_train, Y_train)
        r2s[i] = r2_score(Y_test, classifier_model.predict(X_test))
        costs[i] = mean_squared_error(Y_test, classifier_model.predict(X_test))
    return r2s, costs


# # Importação da classificação "labels" (para todo o conjunto de dados, treinamento e teste)
# y = pd.read_csv('./datasets/Golub/actual.csv')
# y.head()

# # Modificar label para numerico
# y = y.replace({'ALL': -1,'AML':1})
# labels = ['ALL', 'AML'] # para gerar os gráficos(utilizado posteriormente)

# # Importação dos dados de treinamento
# df_train = pd.read_csv('./datasets/Golub/data_set_ALL_AML_train.csv')
# print(df_train.shape)

# # Importação dos dados de teste
# df_test = pd.read_csv('./datasets/Golub/data_set_ALL_AML_independent.csv')
# print(df_test.shape)

# # Remove conunas "call" dos dados de treinamento e teste
# train_to_keep = [col for col in df_train.columns if "call" not in col]
# test_to_keep = [col for col in df_test.columns if "call" not in col]

# X_train_tr = df_train[train_to_keep]
# X_test_tr = df_test[test_to_keep]

# train_columns_titles = ['Gene Description', 'Gene Accession Number', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
#        '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

# X_train_tr = X_train_tr.reindex(columns=train_columns_titles)

# train_columns_titles = ['Gene Description', 'Gene Accession Number', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
#        '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

# X_train_tr = X_train_tr.reindex(columns=train_columns_titles)


# test_columns_titles = ['Gene Description', 'Gene Accession Number','39', '40', '41', '42', '43', '44', '45', '46',
#        '47', '48', '49', '50', '51', '52', '53',  '54', '55', '56', '57', '58', '59',
#        '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72']

# X_test_tr = X_test_tr.reindex(columns=test_columns_titles)


# X_train = X_train_tr.T
# X_test = X_test_tr.T


# # Limpar os nomes das colunas para os dados de treinamento
# X_train.columns = X_train.iloc[1]
# X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

# # Limpar os nomes das colunas para os dados de teste
# X_test.columns = X_test.iloc[1]
# X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)


# # Dividir em treinamento e teste (primeiramente deve-se reiniciar os índices dos conjuntos de dados
# # variando de 0 ao comprimento dos dados -1 como índice).

# # Subset the first 38 patient's cancer types
# X_train = X_train.reset_index(drop=True)
# Y_train = y[y.patient <= 38].reset_index(drop=True)
# Y_train = np.array(Y_train['cancer'])

# # Subset the rest for testing
# X_test = X_test.reset_index(drop=True)
# Y_test = y[y.patient > 38].reset_index(drop=True)
# Y_test = np.array(Y_test['cancer'])

# # Converte de integer para float
# X_train_fl = X_train.astype(float, 64)
# X_test_fl = X_test.astype(float, 64)

# # Aplica a mesma escala para ambos os datasets
# scaler = StandardScaler()
# X_train_scl = scaler.fit_transform(X_train_fl)
# X_test_scl = scaler.transform(X_test_fl) # note that we transform rather than fit_transform

# pca = PCA()
# pca.fit_transform(X_train)

# total = sum(pca.explained_variance_)
# k = 0
# current_variance = 0
# while current_variance/total < 0.9:
#     current_variance += pca.explained_variance_[k]
#     k = k + 1

# print(k, " expressões gênicas correspondem 90% da variância. De 7129 expressões gênicas para ", k, ".", sep='')

# pca = PCA(n_components=k)
# X_train.pca = pca.fit(X_train)
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)

# lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# cv_scores_mean, cv_scores_std = run_cross_validation_on_perceptron(X_train_pca, Y_train, lambdas)

# meanAUC = getMeanAUC_Perceptron()

# clf = Perceptron(penalty='l2', alpha=0)
# score = clf.fit(X_train_pca, Y_train).score(X_test_pca, Y_test)

# print('Executou!')


# Importação dos dados
data = pd.read_excel('./datasets/Concrete_Compressive_Strength/Concrete_Data.xls')
data.head()

# # data information
y = np.array(data.iloc[0:data.shape[0], -1])

X = np.array(data.iloc[0:data.shape[0], 0:-1])

# # Divide os atributos em conjunto de treinamento e de testes na razaõ 75%/ 25%
X_train, X_test, y_train, y_test = train_test_split(X, y)

# # Aplica a mesma escala para ambos os datasets
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test) # note that we transform rather than fit_transform


model = SGDRegressor()
model.fit(X_train, y_train)

r2 = r2_score(y_test, model.predict(X_test_scl))
mse = mean_squared_error(y_test, model.predict(X_test_scl))

# NNeuronios = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# parameters = {'p':NNeuronios, 'lambdar':lambdas}
# model = ELM( classificator = False)

# clf = GridSearchCV(model, parameters, n_jobs = -1, cv = 10)
# clf.fit(X_train_scl, y_train)

np.random.uniform