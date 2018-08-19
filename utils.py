from sympy import sympify, lambdify, var

import netFunctions as nf


def getUserAmount(min, max):
    while True:
        amount = input("Inserisci un numero compreso tra {} e {}\n".format(min, max))
        try:
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
        var('x y')
        while True:
            user_input = input("Definisci una funzione matematica con due variabili x e y\n")
            if "x" in user_input:
                if "y" in user_input:
                    break
                else:
                    print("All'interno della funzione non c'è la variabile y, ricorda che devi inserire sia x che y, "
                          "riprova")
            else:
                print(
                    "All'interno della funzione non c'è la variabile x, ricorda che devi inserire sia x che y, riprova")
        func = lambdify((x, y), sympify(user_input))
    print("Test funzione:")
    if n_variables == 1:
        print("La tua funzione con input 2 da come risultato: {}\n\n".format(func(2)))
    else:
        print("La tua funzione con input x=2 e y=2 da come risultato: {}\n\n".format(func(2, 2)))
    return func


def getActivation(layer):
    print("Che funzione di attivazione vuoi utilizzare nello strato {}?\n"
          "1) Sigmoide\n"
          "2) ReLU\n"
          "3) Tangente iperbolica\n"
          "4) Lineare\n"
          "5) Definita da input\n".format(layer))
    choice = getUserAmount(1, 5)
    if choice == 1:
        f = nf.sigmoid
    elif choice == 2:
        f = nf.ReLU
    elif choice == 3:
        f = nf.tanh
    elif choice == 4:
        f = nf.linear
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
