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

from kerneloptimizer.optimizer import KernelOptimizer


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


class HebbianoWithKernel(BaseEstimator, RegressorMixin):
    """ Classificador Hebbiano"""

    def __init__(self, kernel = 'mlp', classificator = True):
        self.kernel = kernel
        self.W = None
        self.classificator = classificator
        self.opt = None


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

        if self.kernel == 'mlp':
            self.opt = KernelOptimizer(
                kernel='mlp',
                input_dim=X.shape[1],
                hidden_dim=20,
                output_dim=50
            )
        elif self.kernel == 'gaussian':
            self.opt = KernelOptimizer(
                kernel='gaussian'
            )

        self.opt.fit(X, y)

        # Obtem o projeção da entrada no espaço de verossimilhanças
        H = self.opt.get_likelihood_space(X, y)

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

        # Obtem o projeção da entrada no espaço de verossimilhanças
        H = self.opt.get_likelihood_space(X)

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

    def crostalk(self, X, y):

        # Obtem o projeção da entrada no espaço de verossimilhanças
        H = np.array(self.opt.get_likelihood_space(X))

        crostalk = 0

        for k in range(H.shape[0]):
            x_k = H[k,:]
            crostalk += np.sum(np.dot(x_k, np.delete(H.T,k, axis=1)) * np.delete(y, k, axis=0))

        return crostalk/H.shape[0]
            


# Importação da classificação "labels" (para todo o conjunto de dados, treinamento e teste)
data = pd.read_csv('./datasets/parkinsons/parkinsons.data')
data.head()

y = np.array(data.status.replace({0:-1}))

X = data.drop(['status', 'name'], axis=1)

# Divide os atributos em conjunto de treinamento e de testes na razaõ 75%/ 25%
X_train, X_test, y_train, y_test = train_test_split(X, y) 

# Aplica a mesma escala para ambos os datasets
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test) # note that we transform rather than fit_transform

model = HebbianoWithKernel(kernel='mlp').fit(X_train_scl, y_train)

model.crostalk(X_test_scl, y_test)
