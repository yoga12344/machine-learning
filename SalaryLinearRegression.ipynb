{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOWMe6Ba4KsM32cf9LNVIt+",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Abhijith-Reddy-ch/Machine-Learning/blob/main/LinearRegression.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "0rstYC8nY0tF",
        "outputId": "2289d57e-1d6a-4150-b775-113295e286d7"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   YearsExperience   Salary\n",
              "0              1.1  39343.0\n",
              "1              1.3  46205.0\n",
              "2              1.5  37731.0\n",
              "3              2.0  43525.0\n",
              "4              2.2  39891.0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0f1ddbe1-8e95-4144-95ce-121099f3b4b9\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>YearsExperience</th>\n",
              "      <th>Salary</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1.1</td>\n",
              "      <td>39343.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1.3</td>\n",
              "      <td>46205.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1.5</td>\n",
              "      <td>37731.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2.0</td>\n",
              "      <td>43525.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2.2</td>\n",
              "      <td>39891.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0f1ddbe1-8e95-4144-95ce-121099f3b4b9')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-0f1ddbe1-8e95-4144-95ce-121099f3b4b9 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-0f1ddbe1-8e95-4144-95ce-121099f3b4b9');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-421ea77a-8f14-462a-94c1-890a36b93518\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-421ea77a-8f14-462a-94c1-890a36b93518')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-421ea77a-8f14-462a-94c1-890a36b93518 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "data",
              "summary": "{\n  \"name\": \"data\",\n  \"rows\": 30,\n  \"fields\": [\n    {\n      \"column\": \"YearsExperience\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.8378881576627184,\n        \"min\": 1.1,\n        \"max\": 10.5,\n        \"num_unique_values\": 28,\n        \"samples\": [\n          3.9,\n          9.6,\n          3.7\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Salary\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 27414.4297845823,\n        \"min\": 37731.0,\n        \"max\": 122391.0,\n        \"num_unique_values\": 30,\n        \"samples\": [\n          112635.0,\n          67938.0,\n          113812.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 4
        }
      ],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "\n",
        "data=pd.read_csv('/content/salary_data.csv')\n",
        "data.head()\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NXoVdV-cZ7Vy",
        "outputId": "463ed6db-186e-4198-b3f5-4c3491088312"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(30, 2)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 147
        },
        "id": "0WXUlCMsaAyP",
        "outputId": "666f83fe-cb2a-4c7b-8a17-5f014e22dc8d"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "YearsExperience    0\n",
              "Salary             0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>YearsExperience</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Salary</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x = data.iloc[:, :1].values\n",
        "y = data.iloc[:, 1:2].values\n",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "1F-2DW3UaD5j"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model=LinearRegression()\n",
        "model.fit(x_train, y_train)\n",
        "y_pred=model.predict(x_test)"
      ],
      "metadata": {
        "id": "ihlGjuTmaGvH"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_pred)\n",
        "print(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Eyl7_adVainD",
        "outputId": "4428fd24-7ba0-471d-e10e-c8456ad1f3c9"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[115790.21011287]\n",
            " [ 71498.27809463]\n",
            " [102596.86866063]\n",
            " [ 75267.80422384]\n",
            " [ 55477.79204548]\n",
            " [ 60189.69970699]]\n",
            "[[112635.]\n",
            " [ 67938.]\n",
            " [113812.]\n",
            " [ 83088.]\n",
            " [ 64445.]\n",
            " [ 57189.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(x_train, y_train, color='blue')\n",
        "plt.plot(x_train, model.predict(x_train), color='red')\n",
        "plt.title('SALARY VS EXPERIENCE (training set)')\n",
        "plt.xlabel('Experience in Years')\n",
        "plt.ylabel('Salary in Rupees')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "ehEw82yRakwk",
        "outputId": "6bc8e8cb-6ad7-4dd3-8dd9-9990d9ae2cdd"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(x_test, y_test, color='blue')\n",
        "plt.plot(x_train, model.predict(x_train), color='red')\n",
        "plt.title('SALARY VS EXPERIENCE (testing set)')\n",
        "plt.xlabel('Experience in Years')\n",
        "plt.ylabel('Salary in Rupees')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "0KfbuaMaaqH3",
        "outputId": "f0a8dd9a-7cd8-4278-aed5-e017aa9d6de6"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}
