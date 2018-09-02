import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy import sympify, lambdify, var

import netFunctions as nf


def getUserAmount(min, max, Float=False):
    while True:
        amount = input("Inserisci un numero compreso tra {} e {}\n".format(min, max))
        try:
            if Float:
                value = float(amount)
            else:
                value = int(amount)
            if min <= value <= max:
                break
            else:
                print("Il numero dev'essere compreso tra {} e {}, riprova\n".format(min, max))
        except ValueError:
            print("Devi inserire un numero, riprova\n")
    return value


def getUserFunction(n_variables):
    print('\033[93m' + "ATTENZIONE: LA DEFINIZIONE DI UNA FUNZIONE NON VALIDA COMPROMETTERA' L'UTILIZZO DELLA RETE,"
                       " PERTANTO NON SI GARANTISCE IL CORRETTO FUNZIONAMENTO DELLA STESSA" + '\033[0m')
    if n_variables == 1:
        var('x')
        while True:
            user_input = input("Definisci una funzione matematica con una variabile x\n")
            if "x" in user_input:
                break
            else:
                print("All'interno della funzione ci dev'essere la variabile x, riprova")
        func = lambdify(x, sympify(user_input))
    elif n_variables == 2:
        var('t y')
        while True:
            user_input = input(
                "Definisci una funzione di errore con due variabili t e y, dove y indica il valore predetto dalla rete"
                "e t indica il valore che dovrebbe avere\n")
            if "t" in user_input:
                if "y" in user_input:
                    break
                else:
                    print("All'interno della funzione non c'è la variabile y, ricorda che devi inserire sia t che y, "
                          "riprova")
            else:
                print(
                    "All'interno della funzione non c'è la variabile t, ricorda che devi inserire sia t che y, riprova")
        func = lambdify((t, y), sympify(user_input))
    print("Test funzione:")
    if n_variables == 1:
        print("La tua funzione con input 2 da come risultato: {}\n\n".format(func(2)))
    else:
        print("La tua funzione con input t=2 e y=3 da come risultato: {}\n\n".format(func(2, 3)))
    return func


def getActivation(layer):
    print("Che funzione di attivazione vuoi utilizzare nello strato {}?\n"
          "1) Sigmoide\n"
          "2) ReLU\n"
          "3) Tangente iperbolica\n"
          "4) Identità\n"
          "5) Definita da input\n".format(layer))
    choice = getUserAmount(1, 5)
    if choice == 1:
        f = nf.sigmoid
    elif choice == 2:
        f = nf.ReLU
    elif choice == 3:
        f = nf.tanh
    elif choice == 4:
        f = nf.identity
    else:
        f = getUserFunction(1)
    return f


def getErrorFunc():
    print("Che funzione di errore vuoi utilizzare?\n"
          "1) Somma dei quadrati\n"
          "2) Cross Entropy\n"
          "3) Definita da input\n")
    choice = getUserAmount(1, 3)
    if choice == 1:
        f = nf.sum_square
    elif choice == 2:
        f = nf.cross_entropy
    else:
        f = getUserFunction(2)
    return f


def plotGraphErrors(error_t, error_v, title, path=False):
    plt.tight_layout()
    plt.figure(figsize=[10, 10])
    plt.plot(error_t, 'b*')
    plt.plot(error_v, 'r*')
    blue_patch = mpatches.Patch(color='blue', label='ERRORE SU TRAINING SET')
    red_patch = mpatches.Patch(color='red', label='ERRORE SU VALIDATION SET')
    plt.legend(handles=[blue_patch, red_patch])
    plt.ylabel("ERRORE")
    plt.xlabel("EPOCHE")
    plt.title(title)
    if path == False:
        plt.show()
    else:
        plt.savefig(path + title + ".png", format="png")
    plt.close()


def getRightNetResponse(net, data_set):
    right_responses = 0
    for test in data_set:
        out = net.predict(test['input'])
        if test['label'][out.argmax()] == 1:
            right_responses += 1
    return right_responses


def getNumbHiddenLayerRA():
    print("Quanti strati interni vuoi nella rete associativa?\n"
          "1) strato singolo\n"
          "2) 3 strati interni\n")
    choice = getUserAmount(1, 2)
    if choice == 2:
        choice = 3
    return choice
