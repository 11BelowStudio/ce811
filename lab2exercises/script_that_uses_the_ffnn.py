
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

from typing import Tuple

iris_model: keras.Model = keras.models.load_model("ModelIris.h5")

labels: Tuple[str, str, str] = ("Iris Setosa", "Iris Versicolor", "Iris Virginica")


def modelOutputToIndexAndLabel(modelOutput: np.ndarray) -> Tuple[int, str]:

    mo: np.ndarray = modelOutput[0]
    highest_index: int = int(np.where(mo == max(mo))[0])

    return highest_index, labels[highest_index]


def getFlowerInfo(sLen: float, sWid: float , pLen: float, pWid: float) -> Tuple[int, str, np.ndarray]:

    flowerData: np.ndarray = np.array([sLen, sWid, pLen, pWid], ndmin=2)

    modelOutput: np.ndarray = iris_model(flowerData).numpy()

    print(type(modelOutput))
    print(modelOutput)

    index_label: Tuple[int, str] = modelOutputToIndexAndLabel(modelOutput)

    return index_label[0], index_label[1], modelOutput

def formatFlowerInfo(flowerInfo: Tuple[int, str, np.ndarray]) -> str:
    return "FLOWER: {}\nCONFIDENCE: {}\nFULL DATA: {}".format(
        flowerInfo[1],
        flowerInfo[2][0][flowerInfo[0]],
        flowerInfo[2][0]
    )

print(formatFlowerInfo(getFlowerInfo(5.9,3,4.2,1.5)))


print("**********************")
print("* WHAT IRIS IS THAT? *")
print("**********************")
print("a script that should tell you what iris you're looking at")

keepGoing: bool = True

while keepGoing:

    print("")
    print("enter the following details about the iris")
    print("(enter anything that isn't a number if you want to quit)")
    print("")

    try:
        sepalLen: float = float(input("Sepal Length (cm): "))
        sepalWid: float = float(input("Sepal width (cm): "))
        petalLen: float = float(input("Petal length (cm): "))
        petalWid: float = float(input("Petal width (cm): "))

        fullData: Tuple[int, str, np.ndarray] = getFlowerInfo(sepalLen, sepalWid, petalLen, petalWid)

        print("")
        print(formatFlowerInfo(fullData))

        print("")

    except ValueError as e:
        print(e)

        print("\nWelp, that wasn't a number, so I guess you're done.")
        keepGoing = False

