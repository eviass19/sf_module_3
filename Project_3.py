{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn import metrics\n",
    "from sklearn import preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The original dataset \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
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
       "      <th>Restaurant_id</th>\n",
       "      <th>City</th>\n",
       "      <th>Cuisine Style</th>\n",
       "      <th>Ranking</th>\n",
       "      <th>Rating</th>\n",
       "      <th>Price Range</th>\n",
       "      <th>Number of Reviews</th>\n",
       "      <th>Reviews</th>\n",
       "      <th>URL_TA</th>\n",
       "      <th>ID_TA</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>id_5569</td>\n",
       "      <td>Paris</td>\n",
       "      <td>['European', 'French', 'International']</td>\n",
       "      <td>5570.0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>$$ - $$$</td>\n",
       "      <td>194.0</td>\n",
       "      <td>[['Good food at your doorstep', 'A good hotel ...</td>\n",
       "      <td>/Restaurant_Review-g187147-d1912643-Reviews-R_...</td>\n",
       "      <td>d1912643</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>id_1535</td>\n",
       "      <td>Stockholm</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1537.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10.0</td>\n",
       "      <td>[['Unique cuisine', 'Delicious Nepalese food']...</td>\n",
       "      <td>/Restaurant_Review-g189852-d7992032-Reviews-Bu...</td>\n",
       "      <td>d7992032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>id_352</td>\n",
       "      <td>London</td>\n",
       "      <td>['Japanese', 'Sushi', 'Asian', 'Grill', 'Veget...</td>\n",
       "      <td>353.0</td>\n",
       "      <td>4.5</td>\n",
       "      <td>$$$$</td>\n",
       "      <td>688.0</td>\n",
       "      <td>[['Catch up with friends', 'Not exceptional'],...</td>\n",
       "      <td>/Restaurant_Review-g186338-d8632781-Reviews-RO...</td>\n",
       "      <td>d8632781</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>id_3456</td>\n",
       "      <td>Berlin</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3458.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3.0</td>\n",
       "      <td>[[], []]</td>\n",
       "      <td>/Restaurant_Review-g187323-d1358776-Reviews-Es...</td>\n",
       "      <td>d1358776</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>id_615</td>\n",
       "      <td>Munich</td>\n",
       "      <td>['German', 'Central European', 'Vegetarian Fri...</td>\n",
       "      <td>621.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>$$ - $$$</td>\n",
       "      <td>84.0</td>\n",
       "      <td>[['Best place to try a Bavarian food', 'Nice b...</td>\n",
       "      <td>/Restaurant_Review-g187309-d6864963-Reviews-Au...</td>\n",
       "      <td>d6864963</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Restaurant_id       City                                      Cuisine Style  \\\n",
       "0       id_5569      Paris            ['European', 'French', 'International']   \n",
       "1       id_1535  Stockholm                                                NaN   \n",
       "2        id_352     London  ['Japanese', 'Sushi', 'Asian', 'Grill', 'Veget...   \n",
       "3       id_3456     Berlin                                                NaN   \n",
       "4        id_615     Munich  ['German', 'Central European', 'Vegetarian Fri...   \n",
       "\n",
       "   Ranking  Rating Price Range  Number of Reviews  \\\n",
       "0   5570.0     3.5    $$ - $$$              194.0   \n",
       "1   1537.0     4.0         NaN               10.0   \n",
       "2    353.0     4.5        $$$$              688.0   \n",
       "3   3458.0     5.0         NaN                3.0   \n",
       "4    621.0     4.0    $$ - $$$               84.0   \n",
       "\n",
       "                                             Reviews  \\\n",
       "0  [['Good food at your doorstep', 'A good hotel ...   \n",
       "1  [['Unique cuisine', 'Delicious Nepalese food']...   \n",
       "2  [['Catch up with friends', 'Not exceptional'],...   \n",
       "3                                           [[], []]   \n",
       "4  [['Best place to try a Bavarian food', 'Nice b...   \n",
       "\n",
       "                                              URL_TA     ID_TA  \n",
       "0  /Restaurant_Review-g187147-d1912643-Reviews-R_...  d1912643  \n",
       "1  /Restaurant_Review-g189852-d7992032-Reviews-Bu...  d7992032  \n",
       "2  /Restaurant_Review-g186338-d8632781-Reviews-RO...  d8632781  \n",
       "3  /Restaurant_Review-g187323-d1358776-Reviews-Es...  d1358776  \n",
       "4  /Restaurant_Review-g187309-d6864963-Reviews-Au...  d6864963  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('main_task.csv')\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Let's change the names of the columns for own convenience and look into data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns=['rest_id','city','cuisine_style','ranking','rating','price_range','number_of_reviews','reviews','url_ta','id_ta']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 40000 entries, 0 to 39999\n",
      "Data columns (total 10 columns):\n",
      "rest_id              40000 non-null object\n",
      "city                 40000 non-null object\n",
      "cuisine_style        30717 non-null object\n",
      "ranking              40000 non-null float64\n",
      "rating               40000 non-null float64\n",
      "price_range          26114 non-null object\n",
      "number_of_reviews    37457 non-null float64\n",
      "reviews              40000 non-null object\n",
      "url_ta               40000 non-null object\n",
      "id_ta                40000 non-null object\n",
      "dtypes: float64(3), object(7)\n",
      "memory usage: 3.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### We can see quite a lot of missed values at cuisine_style & price_range & number_of_reviews / will fill it later"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "rest_id              11909\n",
       "city                    31\n",
       "cuisine_style         9007\n",
       "ranking              11936\n",
       "rating                   9\n",
       "price_range              3\n",
       "number_of_reviews     1459\n",
       "reviews              33516\n",
       "url_ta               39980\n",
       "id_ta                39980\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### - As we can see, restaurant_id is not a unique identificator of the restaurant, but smth else. May be chain of the restaurants has one id for all it's objects or it's just an id into the particular city/area\n",
    "#### - Also rest_id and ranking have almost same qty of unique values, we should check the correlation between these atributes\n",
    "#### - url_ta and id_ta have the biggest qty of unique values into entire dataset, so we can treat it as possible identificator. Also these two attributes have no missed values but 20 unique values less. That may mean there are some duplicates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.rest_id = df.rest_id.str.replace('id_', '').astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 40000 entries, 0 to 39999\n",
      "Data columns (total 10 columns):\n",
      "rest_id              40000 non-null float64\n",
      "city                 40000 non-null object\n",
      "cuisine_style        30717 non-null object\n",
      "ranking              40000 non-null float64\n",
      "rating               40000 non-null float64\n",
      "price_range          26114 non-null object\n",
      "number_of_reviews    37457 non-null float64\n",
      "reviews              40000 non-null object\n",
      "url_ta               40000 non-null object\n",
      "id_ta                40000 non-null object\n",
      "dtypes: float64(4), object(6)\n",
      "memory usage: 3.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
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
       "      <th>rest_id</th>\n",
       "      <th>ranking</th>\n",
       "      <th>rating</th>\n",
       "      <th>number_of_reviews</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>rest_id</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.368308</td>\n",
       "      <td>-0.222637</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>ranking</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.368371</td>\n",
       "      <td>-0.222670</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>rating</td>\n",
       "      <td>-0.368308</td>\n",
       "      <td>-0.368371</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.030964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>number_of_reviews</td>\n",
       "      <td>-0.222637</td>\n",
       "      <td>-0.222670</td>\n",
       "      <td>0.030964</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                    rest_id   ranking    rating  number_of_reviews\n",
       "rest_id            1.000000  1.000000 -0.368308          -0.222637\n",
       "ranking            1.000000  1.000000 -0.368371          -0.222670\n",
       "rating            -0.368308 -0.368371  1.000000           0.030964\n",
       "number_of_reviews -0.222637 -0.222670  0.030964           1.000000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The correlation between rest_id and ranking is 1.0, so we should drop rest_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(columns=['rest_id'], inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop_duplicates(subset=['url_ta'], inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 39980 entries, 0 to 39999\n",
      "Data columns (total 9 columns):\n",
      "city                 39980 non-null object\n",
      "cuisine_style        30701 non-null object\n",
      "ranking              39980 non-null float64\n",
      "rating               39980 non-null float64\n",
      "price_range          26101 non-null object\n",
      "number_of_reviews    37437 non-null float64\n",
      "reviews              39980 non-null object\n",
      "url_ta               39980 non-null object\n",
      "id_ta                39980 non-null object\n",
      "dtypes: float64(3), object(6)\n",
      "memory usage: 3.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "London        5757\n",
      "Paris         4897\n",
      "Madrid        3088\n",
      "Barcelona     2734\n",
      "Berlin        2155\n",
      "Milan         2133\n",
      "Rome          2078\n",
      "Prague        1443\n",
      "Lisbon        1300\n",
      "Vienna        1166\n",
      "Amsterdam     1086\n",
      "Brussels      1060\n",
      "Hamburg        949\n",
      "Munich         893\n",
      "Lyon           892\n",
      "Stockholm      820\n",
      "Budapest       816\n",
      "Warsaw         727\n",
      "Dublin         673\n",
      "Copenhagen     659\n",
      "Athens         628\n",
      "Edinburgh      596\n",
      "Zurich         538\n",
      "Oporto         513\n",
      "Geneva         481\n",
      "Krakow         443\n",
      "Oslo           385\n",
      "Helsinki       376\n",
      "Bratislava     301\n",
      "Luxembourg     210\n",
      "Ljubljana      183\n",
      "Name: city, dtype: int64\n",
      "31\n"
     ]
    }
   ],
   "source": [
    "print(df.city.value_counts())\n",
    "\n",
    "print(df.city.nunique())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Cities.\n",
    "#### There are 31 cities and majority of the restaurants are located in Paris and London"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1cc76f63a88>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAEwCAYAAABbv6HjAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO2debgdRbW331/CaCJhkhgSIICAMoiagCAihEFQRJBBiaCgKIpcBkWR6P1EUQRFFEHgCjIpSAwXkDAJCAnITMIUpghCGAThKoOJDBpY3x+rOqdPn9579z5zTq/3efrZu6tXV1VPq6tXrVolMyMIgiCoB8MGugJBEARB/xFKPwiCoEaE0g+CIKgRofSDIAhqRCj9IAiCGhFKPwiCoEZUUvqSlpf0v5IelvSQpM0lrSjpWkmPpN8VcvJTJD0qaa6kHXLpEyTNSdtOkqS+OKggCIKgnKot/Z8DfzCzdwIbAw8BRwLXmdk6wHVpHUnrA3sBGwA7AqdKGp7yOQ04AFgnLTv20nEEQRAEFVCrwVmSlgPuBdaynLCkucDWZvaspDHATDNbT9IUADM7NsldDXwXmAfMSC8OJE1O+3+pWfkrr7yyjR8/vlsH969//YsRI0YMWrnFoY5xbgaf3OJQxzg3/SfXiNmzZ//dzN7WZYOZNV2A9wB3AOcAdwO/AkYALxXkXky/vwD2yaWfCewBTAT+mEvfEri8VfkTJkyw7jJjxoxBLTeQZQ92uYEse7DLDWTZg11uIMseyGMuA5hlJTq1Skt/InAbsIWZ3S7p58A/gYPNbPmc3ItmtoKkU4Bbzey8lH4mcCXwJHCsmW2X0rcEjjCznUvKPAA3AzF69OgJU6dObVrHRixYsICRI0cOWrnFoY5xbgaf3OJQxzg3/SfXiEmTJs02s4ldNpS9CaxzC/7twDzr3EK/ApgLjElpY4C56f8UYEpO/mpg8yTzcC59MvDLVuVHS7+ecgNZ9mCXG8iyB7vcQJa9uLT0W3bkmtnfgKckrZeStgUeBKYD+6a0fYFL0//pwF6Slpa0Jt5he4eZPQvMl7RZ8tr5bG6fIAiCoB9YoqLcwcD5kpYCHgM+h3v+TJO0P2662RPAzB6QNA1/MSwEDjKzN1I+B+J9A8sCV6UlCIIg6CcqKX0zuwfviC2ybQP5Y4BjStJnARu2U8EgCIKg94gRuUEQBDUilH4QBEGNCKUfBEFQI0LpB0EQ1Iiq3jsBMOevL7PfkVc0lZl33E79VJsgCIL2iZZ+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCMqKX1J8yTNkXSPpFkpbUVJ10p6JP2ukJOfIulRSXMl7ZBLn5DyeVTSSZLU+4cUBEEQNKKdlv4kM3uPmU1M60cC15nZOsB1aR1J6wN7ARsAOwKnShqe9jkNOABYJy079vwQgiAIgqr0xLyzC3Bu+n8usGsufaqZvW5mjwOPAptKGgMsZ2a3mpkBv87tEwRBEPQDVZW+AddImi3pgJQ22syeBUi/q6T0scBTuX2fTmlj0/9iehAEQdBPyBvdLYSkVc3sGUmrANcCBwPTzWz5nMyLZraCpFOAW83svJR+JnAl8CRwrJltl9K3BI4ws51LyjsANwMxevToCVOnTu3WwS1YsICRI0f2mtzzL7zMc682l9lo7KjK+fVFHYeK3OJQxzg3g09ucahjXxxzGZMmTZqdM8d3YGZtLcB3ga8Dc4ExKW0MMDf9nwJMyclfDWyeZB7OpU8GftmqvAkTJlh3mTFjRq/KnXTe722Nb17edGknv76o41CRG8iyB7vcQJY92OUGsuyBPOYygFlWolNbmnckjZD01uw/8GHgfmA6sG8S2xe4NP2fDuwlaWlJa+IdtneYm4DmS9osee18NrdPEARB0A8sUUFmNHBJ8q5cAvitmf1B0p3ANEn746abPQHM7AFJ04AHgYXAQWb2RsrrQOAcYFngqrQEQRAE/URLpW9mjwEbl6T/A9i2wT7HAMeUpM8CNmy/mt1jzl9fZr8jr2gpd86OI/qhNkEQBANPjMgNgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghpRWelLGi7pbkmXp/UVJV0r6ZH0u0JOdoqkRyXNlbRDLn2CpDlp20mS1LuHEwRBEDSjnZb+ocBDufUjgevMbB3gurSOpPWBvYANgB2BUyUNT/ucBhwArJOWHXtU+yAIgqAtKil9SeOAnYBf5ZJ3Ac5N/88Fds2lTzWz183sceBRYFNJY4DlzOxWMzPg17l9giAIgn6gakv/ROAI4M1c2mgzexYg/a6S0scCT+Xknk5pY9P/YnoQBEHQT8gb3U0EpI8BHzWzr0jaGvi6mX1M0ktmtnxO7kUzW0HSKcCtZnZeSj8TuBJ4EjjWzLZL6VsCR5jZziVlHoCbgRg9evSEqVOnduvgnn/hZZ57tbXcmqOGM3LkyF7Jb6Oxo1iwYEGl/IDKsnWTWxzqGOdm8MktDnXsi2MuY9KkSbPNbGKXDWbWdAGOxVvl84C/Aa8A5wFzgTFJZgwwN/2fAkzJ7X81sHmSeTiXPhn4ZavyJ0yYYN3lpPN+b2t88/KWy4wZM3otPzOrnF87snWTG8iyB7vcQJY92OUGsuyBPOYygFlWolNbmnfMbIqZjTOz8XgH7fVmtg8wHdg3ie0LXJr+Twf2krS0pDXxDts7zE1A8yVtlrx2PpvbJwiCIOgHlujBvscB0yTtj5tu9gQwswckTQMeBBYCB5nZG2mfA4FzgGWBq9ISBEEQ9BNtKX0zmwnMTP//AWzbQO4Y4JiS9FnAhu1WMgiCIOgdYkRuEARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIlkpf0jKS7pB0r6QHJH0vpa8o6VpJj6TfFXL7TJH0qKS5knbIpU+QNCdtO0mS+uawgiAIgjKWqCDzOrCNmS2QtCRwk6SrgN2A68zsOElHAkcC35S0PrAXsAGwKvBHSeua2RvAacABwG3AlcCOwFW9flTBYsn4I6/otH74RgvZr5AGMO+4nfqrSkEw5GjZ0jdnQVpdMi0G7AKcm9LPBXZN/3cBpprZ62b2OPAosKmkMcByZnarmRnw69w+QRAEQT9QyaYvabike4DngWvN7HZgtJk9C5B+V0niY4Gncrs/ndLGpv/F9CAIgqCfkDe6KwpLywOXAAcDN5nZ8rltL5rZCpJOAW41s/NS+pm4KedJ4Fgz2y6lbwkcYWY7l5RzAG4GYvTo0ROmTp3arYN7/oWXee7V1nJrjhrOyJEjeyW/jcaOYsGCBZXyAyrL1kFuzl9f7rQ+ellKz/dGY0cNWB0Hg9ziUMc4N/0n14hJkybNNrOJxfQqNv1FmNlLkmbitvjnJI0xs2eT6eb5JPY0sFput3HAMyl9XEl6WTmnA6cDTJw40bbeeut2qrmIk8+/lBPmtD7Ec3YcQZUyquQ3b++tmTlzZqX8gMqydZAr2u8P32hh6fmet3fn/QfjsfSl3ECWPdjlBrLsgTzmdqjivfO21MJH0rLAdsDDwHRg3yS2L3Bp+j8d2EvS0pLWBNYB7kgmoPmSNkteO5/N7RMEQRD0A1Va+mOAcyUNx18S08zsckm3AtMk7Y+bbvYEMLMHJE0DHgQWAgclzx2AA4FzgGVxr53w3AmCIOhHWip9M7sPeG9J+j+AbRvscwxwTEn6LGDD9qsZBEEQ9AYxIjcIgqBGhNIPgiCoEaH0gyAIakQo/SAIghrRlp9+EHSHfEydRvF0giDoH6KlHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjQilHwRBUCNC6QdBENSIUPpBEAQ1IpR+EARBjYiJ0XuZ8UdeUWny73nH7dRPNQqCIOgglP4QYXx6ybR64cTLJgjqTUvzjqTVJM2Q9JCkByQdmtJXlHStpEfS7wq5faZIelTSXEk75NInSJqTtp0kSX1zWEEQBEEZVVr6C4HDzewuSW8FZku6FtgPuM7MjpN0JHAk8E1J6wN7ARsAqwJ/lLSumb0BnAYcANwGXAnsCFzV2wcV9A/jc18UVUxaQRAMPC1b+mb2rJndlf7PBx4CxgK7AOcmsXOBXdP/XYCpZva6mT0OPApsKmkMsJyZ3WpmBvw6t08QBEHQD7TlvSNpPPBe4HZgtJk9C/5iAFZJYmOBp3K7PZ3Sxqb/xfQgCIKgn5A3uisISiOBG4BjzOxiSS+Z2fK57S+a2QqSTgFuNbPzUvqZuCnnSeBYM9supW8JHGFmO5eUdQBuBmL06NETpk6d2q2De/6Fl3nu1dZya44azsiRI3stv9HL0lJuo7GjAFiwYEGlslvJzfnry5XK7s1yszKrlNuuXDPZ7Biq1HEoyi0OdYxz039yjZg0adJsM5tYTK/kvSNpSeAi4HwzuzglPydpjJk9m0w3z6f0p4HVcruPA55J6eNK0rtgZqcDpwNMnDjRtt566yrV7MLJ51/KCXNaH+I5O46gShlV8zt8o4Ut5ebt7eXNnDmzUtmt5PbLee80K7s3y92vYNPvrXPTSjY7hip1HIpyA1n2YJcbyLIH8pjboYr3joAzgYfM7Ke5TdOBfdP/fYFLc+l7SVpa0prAOsAdyQQ0X9JmKc/P5vYJgiAI+oEqTa4tgM8AcyTdk9K+BRwHTJO0P2662RPAzB6QNA14EPf8OSh57gAcCJwDLIt77YTnThAEQT/SUumb2U1AI3/6bRvscwxwTEn6LGDDdioYBEEQ9B4ReycIgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakQo/SAIghoRSj8IgqBGhNIPgiCoEaH0gyAIakS1eeuCIAgKjM9Nl1nk8I0WLppOc95xO/VXlYIKREs/CIKgRoTSD4IgqBGh9IMgCGpEKP0gCIIaER25QRB0omoHbbB4Ei39IAiCGhFKPwiCoEaE0g+CIKgRofSDIAhqRHTkBl3Id+RFx10QDC2ipR8EQVAjWip9SWdJel7S/bm0FSVdK+mR9LtCbtsUSY9Kmitph1z6BElz0raTJKn3DycIgiBoRhXzzjnAL4Bf59KOBK4zs+MkHZnWvylpfWAvYANgVeCPktY1szeA04ADgNuAK4Edgat660CCYCjSyGc+ApoF3aWl0jezGyWNLyTvAmyd/p8LzAS+mdKnmtnrwOOSHgU2lTQPWM7MbgWQ9GtgV0LpB31IUWGW9U+EwgzqRndt+qPN7FmA9LtKSh8LPJWTezqljU3/i+lBEARBPyIzay3kLf3LzWzDtP6SmS2f2/6ima0g6RTgVjM7L6WfiZtyngSONbPtUvqWwBFmtnOD8g7ATUGMHj16wtSpU7t1cM+/8DLPvdpabs1Rwxk5cmSv5Td6WVrKbTR2FAALFiyoVHYruTl/fblS2VXKzfKqkl9fyTWTzY4ho9Gx5I+jUX7FvJrlNxByxWPIyB9L2TH0pOxGZRbLbUZf1q+3ZIeKXCMmTZo028wmFtO767L5nKQxZvaspDHA8yn9aWC1nNw44JmUPq4kvRQzOx04HWDixIm29dZbd6uSJ59/KSfMaX2I5+w4giplVM3v8I0WtpSbt7eXN3PmzEplt5LLzBatyq5S7n4Fl83eOuZ25JrJZseQ0ehYiqacsvyKeTXLbyDkGrnL5o+l7Bh6UnYzF93uXOferl9vyQ4VuXbprtKfDuwLHJd+L82l/1bST/GO3HWAO8zsDUnzJW0G3A58Fji5RzUPgqAtIpBaABWUvqQL8E7blSU9DRyFK/tpkvbHTTd7ApjZA5KmAQ8CC4GDkucOwIG4J9CyeAdudOIGQRD0M1W8dyY32LRtA/ljgGNK0mcBG7ZVuyHM+Jw5plkLK7xLglZECz5ohxiRGwRBUCMi9g7urVClNXT4Rv1QmWCxpqzVHeMDgsFEtPSDIAhqRCj9IAiCGhFKPwiCoEaETT8IKlDVVh8Eg51o6QdBENSIaOkHix1VomcOduLLIRgoQukHQRC0oNkAuIzDN1q4KN78YCaUfhAEiw1DSfkOFGHTD4IgqBHR0g9qTdjWg7oRSn+QUzUwWxAEQRVC6QdBMOQI239jwqYfBEFQI6KlHwRB0I9Ujep7zo4j+qT8aOkHQRDUiGjpB0HQp/TGzF7hyNB7REs/CIKgRoTSD4IgqBFh3qkZ4fcfBPUmWvpBEAQ1IpR+EARBjQjzThAEQS9RbSRwP1SkCaH0gyCoLVXDNQwlVRnmnSAIghoRSj8IgqBG9LvSl7SjpLmSHpV0ZH+XHwRBUGf6VelLGg6cAnwEWB+YLGn9/qxDEARBnenvlv6mwKNm9piZ/RuYCuzSz3UIgiCoLf2t9McCT+XWn05pQRAEQT8gM+u/wqQ9gR3M7Atp/TPApmZ2cEHuAOCAtLoeMLebRa4M/H0Qyw1k2YNdbiDLHuxyA1n2YJcbyLIH8pjLWMPM3tYl1cz6bQE2B67OrU8BpvRhebMGs9ziUMc4N4NPbnGoY5yb/j3mdpb+Nu/cCawjaU1JSwF7AdP7uQ5BEAS1pV+HmZnZQkn/BVwNDAfOMrMH+rMOQRAEdabfxxab2ZXAlf1U3OmDXG4gyx7scgNZ9mCXG8iyB7vcQJY9kMdcmX7tyA2CIAgGlgjDEARBUCNC6QdBENSIUPpB0AdIWrpKWhD0N0MnSHRC0gjgVTN7U9K6wDuBq8zsPwW54cBOwHhy58HMftqDsj9Qkt+vG8iOBdYoyN5YkNkY2DKt/snM7u1u3XJ5rgCsAyzTqNw28hKwN7CWmR0taXXg7WZ2R0/r2VtI2gK4x8z+JWkf4H3Az83siYJcletxNPAn4BYz+1eLom9NZbVKy+7F0YWyn2xwPMsV5F5oUY+GSNqtJPllYI6ZPV+QrXRvS/ogsI6ZnS3pbcBIM3u8RG40sElavaNYXneoUkdJOwEb0Pn+P7pBfu3IrgCsZmb3Ndje5brj5/oJM1uYk/sTcCN+n91sZvPL8usJQ07p4ydsy3QRrgNmAZ/ClVOey4DXgDnAm40ySzfuF+l6M32+IPcbYG3gHuCNTAwoezB+lOr0YEH2xpzMoanci1PSeZJON7OTS/LbDfgRsAqgtJiZLVeQ+wJwKDAu1XMzXBFtk5NZBtifrjd7p+NNnIqfu22Ao4H5wEV0PMxZnpsBJwPvApbC3XX/la+fpJPTOSjFzA4p5FnpugCnARunF+gRwJn4Ndkql1fL65GYB0wGTpI0H38wbzSzS3N5vR0PLbKspPfi1wJgOeAtxeOSdDBwFPAcHfehAe8uyH0JP8ev0nGeDFirJM9K9wN+nTcHZqT1rYHbgHUlHW1mv0n5Vbq3JR0FTMRH0Z8NLAmcB2xRkPskcDwwM9XtZEnfMLP/TdtPNLPDJF1GyT1hZh8vOeaWdZT0P/g1mAT8CtgDKG2gVJGVNBP4OH7/3QP8n6QbzOxrJVmeir/w70vHvGH6v5KkL5vZNUluX+CDwO7A8ZJexxt8Xy2rZ7foixFfA7kAd6Xfg4Ej0v+7S+Tuq5jfLfgD9Ml0IXYHdi+Re4jkDVUhz7nA0i1k7gNG5NZHNKoz8CjwrgrlzsEV+T1p/Z3A7woyFwLfB/6SbsBr8JZxs3N9dy7t3hK5WcA7gLtxhf854JiCzL5pOR24KV2/g3HF+7MeXJesjt8B9s+ntXM9CvJvBw4BngTmlxzHDPwFeH36PwMfhLhbg2u3UoUyHwFWrli/qvfDZcDo3PpovJGxInB/u/c2rvhUuB+63LPAvcAqufW35e8bYEL63apsaVB2yzpmdcn9jgSu6a5sdpzAF4DvNTrelD4V2CC3vj7+YlyL9Dzmto3BB66egjdE/lD13qyyDMWWviRtjrfs909pZcd5laQPW8cbthFvMbNvVij3flwZPFtB9jG8FfR6ExnR0WIh/VcD2efM7KEK5b5mZq9JQtLSZvawpPUKMu8wsz0l7WJm50r6LT6Yroz/JNOEwaLWd+lXk5k9Kmm4mb0BnC3plsL2c1Me+wGTLJnjUour7BpVvS7zJU0B9gE+lOq7ZEGmyvVA0q/wh/U5vJW/B3BXyXGcK2l3M7uoQv2ewj/zW/EX4JUKclD9fhhvZs/l1p8H1jWzFyTlzaFV7+1/m5lJyu6HEQ3khllnc84/yPUvmtns9HdB7j8pz50b5Fmljq+m31ckrZrKXbMHsktIGoM3PL7dpFyAd1puIKqZPSjpvWb2mFtJHUl/wePt/Bb/Kj3YzBpaIrrDUFT6h+ExfS4xswckrUXH52ue24BLJA0D/kPjT+DLJX3UfFBZM1YGHpR0BznlYblP0ZwJ4xXgHknXFWTzJoyzgdslXZLWdwXOalD2LEm/A35fyO/igtzTkpZPctdKehF4piCTPewvSdoQ+BtuQinjJOASYBVJx+BK8L9L5F5JYTfukfRj/MFspBBWBd4KZLbqkSmtSNXr8ing03gr/2+p3+F4aPt6AKyEf6m8lOr3d8vZYwuMS/b3+cAZ+Kf9kSWNjMeAmZKuKJRd7FuaAtwi6fYWdYTq98OfJF2Of92Bfy3dmJT1Szm5lvd2YpqkXwLLS/oi8Pl07EX+IOlq4IK0/inKB2yeIWlfM5sDIGky/nxflgnkTEBvrVDHy9P9fzz+srYG9Wsk+6uCzNF4g+gmM7sz6ZpHGuT3Z0mn4S3+7Jj/nDr38y/Yk3DzzmTgvcANkm40s780yLdtajs4S9JjuCKdY01OQrLdjgD+TcfF6fJykLRVcd8keENOZt9mdcpauzn59+E3gHDb8d0N6nh2eXaldvh8fUfhn47/zqV/AbfLvxt/8YwEvmNm/9Mgn3cC26Y6XlfWwpS0Bt46Xgr4air3VDN7tET2c8B36XhRbwV8t+TcZNfldZq/tBvS7vXI7fcuYId0LMPNbFyJzL1mtrGkHYCDgP8HnG1m7yvIHdWg7O8V5O7AzV6d+qDK6tjO/SBpdzrusZuAi4rPQ5V7Oye7PfDhlN/VZnZt2b6p3C3ouLcvKZFZC/hf/Kv9g8BngY+Z2cs5mdK6Natj2m9pYJl8Xo1oR7ZJHssCX6HzuT4V71d8i5ktKMiPxM2gXwfGmdnw7pbdpS5DRem32/mTWhof6e1PpyqkltRrydSReW8sbWav5GR+Y2afKezXJa3Ncn+O2/BvaSlcPc8VgNXo3Jl6V+M9KuX5duD9afV2M/tbN/KYT3nHcJeXQ5XrkdI/hntTfQhYAe8E/5OZdfkCk3Sfmb07nfOZZnaJpLvN7L0N6jvCmngESbrFzD7Q4rArk75w7zOzDSvK97q3TcVy18W/WJ4CdjWzV1vs0iyve4Hf4c9Ay5azWngDqbqTx3D8BbhdhTJPwF8MI3FrxI34PfZYq32rMpTMO79Jvz+pKP8s/ll9Fc0/q5H0cfxBB3+AL89tu8nMPliiZJq1PK8DtgOyt/uyuN06/1BvUKjDcGBC2YGousfNXcB/pwfpEvzmn1XIa2n8M388nW/kLq5qkr4P7Ifbm/MeJdsU5LbAW+9Fl8i1cjLvTH0MWUs4m2xnVUmrZi+SErlOZHJm9tay7Q2ocj3Ap/m8Ee/YLprFisyWdA1uB54i6a2U9HfI+5/OxB/y1eVeRl8ys68URGfI55m4jM73axeXTUnjcG+pLfDrcRNwqJk9ndvvTUn3SlrdGriH5vJr6m2Tk8s/A0vh/SSLvLRy20WTZ0XSnML2FXGz2u2SMLNOnk0lZWe8jDsRHJ6U5sdxs8o0SW/iL4BpZcevah5Ll+J9O3+kc/9bJ8zsDUmvSBpV4WvhNuDHhb6WXmXItPRhkWI818z2qSBb9bP6OLyFc35KmgzMNrNuT+ou6R4ze09ZmrzT8Vu44slamsLNS6eb2ZSS/C4EHsZt10fjn8MPmdmhDcpfEVfsewGrm9k6uW1/wB+W2eRuZDM7oSSfucBGefNQg/Iexk0hxTz/kZM53cwOkFTW/2Jmtk2SO8PMvthKrlB+3nd8ZeCtlvMdb3Y9mh1XM1JL+j3AY2b2kqSVgLFW8OOW2+j3AKZnXwGS7i+2wCV18XXHj7fMZfNavCMwawjtA+xtZtsX5K7H7+07gEVfGSVfxfcC22et+9TC/aOZbdziHOyKT5L0rWZyJfut0Wy7FcZYpH2+h/dP/RZ/XvbCO3bnAgea2dYF+XVwk9veZaYTSQ8B6xdNXQWZyveIpGm4i/S1dD7XXfpkCo3MG8zssqJMj7BedAUaDAvesbJUL+Z3H+5tkK0PJ+eWhbdCGi4N8rwZeF9ufQJwa259GB52umodM9exzL1sSeD6JvKbAifgLfTLCtvub6Pci8i53jWRu71ifsOALXr5fjgKbx3/Oa2vig96qXw9cumb4XNCLMBfwm8A/2xQrnBl+520vjquAEvPDS3cXts85nsqplVyicT7vYrXaU7FutxWkrY2yUUWHxtwCLB8g/2Hp2u2erZUvU1QkyEAACAASURBVMeysunsDjoeH68xG3/ZHd4gvwuBMS2O7QfARyueh33LlhK5Y/Evz8+n5Vrg2N54FrJlKJl3MuYBN0uaTuc36k+hU29/KVYy8ANYng5vklGFbbPp+GTtkh0lg2dwD4QLJWUmgjH4Z2dWhzfTZ35VKnncyAch7YYr+2nA983spYLYLZI2suQx0YJjgbsl3U9zr44Zko7HfcDzckV3xzcl/QQfMFSKykeR5vMoeqh8AveCyMw+zyRTS56m1yPHL/AW5IX4IKTP4uMPyqg0cA14KtmOTe7hdAjuc96FdG3Xp7MJr2zE99/lo48z75jJuMthJ6xBJ2cJlbxtCtdmGH6Oyp61i4CJkt6Bm7am4y30jxbyqzRwLfFmMkNlJqc9ctsyF9Lb8QbRhcCe1txOXsVj6VDgW5L+jTcCGpp0rYFTQAk7Ae+x1Nco6Vx8fEuXL/zuMhSV/jNpGYa7cRWpavPPyBTbDPyifojcBTCzNSUJH4Ld1Daa2+dOudfLeinPh60QJgK4TdImZnZnhSxPTx2q/w9/gEbig5GKPA5sbmbN5t38ILBfMie8TseNXPagnYsPkGo6qpmOTtmJubQutv/ENXLPjostNX0KNPLTzvIsKv2WvuMVr0cm23S8QY73m9n7JN2d9nsxKfUiXwZ+jo/ifRrvSzioKJTMkVvjSv9KvH/hJkpGfOMtxF8AP8PPyS0prZhnUxt87pi/kRR65nlyupV429D52izEG2C7lMi9aT6h0m7AiWZ2cnaeChwKrGc5M2AT9sbP46npmG4D9pF7zfxXktnXzB6ukBd4H1RTrI1+o/Q8lTmYlDUKmzUye8yQsum3S3oI102rcxs96PIBGJvgN3ypN4mk2WZW2tGak9nGzK5v1FrNt1IlPZjq9gT+xdJM+VZGLWLvNLKnWrkd9QYza+oy1436Za6YC3F3trZdMQv5fR0/3u3xF/jngd8mRVP5eqS8bsQ7fH+Ff009C+xnJbbt1Kr8AHBnUv5vw0d0lnrvVDiOOcDGuBloY7k3za/MrMtLUNKK1o2YPM1s8OrwqHoTP6a2Papyed0OnIgPaNrZzB5v0I8xA+9LaDQWIpMbDhxnZt9oITca+CGwqpl9RNL6eCPozG4eRxZ7ak0z+76k1XCTUJfQDqlPJ2MZYE/c/Pudgtxk4DjcZXlRI9PMptJLDDmlnx6uI+jqyVL0KNkab6nOw0/uanhL4Ma0vdQ7JJdfJ9OEpFOAc5q1zCV9z8yOUgU/6irKV1JZjI+8bCdPJDWIvVNybioFAJP0U/xrYDpNzDaSRuGf6Ys6p4CjrWd+zyulPD9Ih4fK0WWtQjXwHW/neiT5dsYb7I2bQSYA55AGrpnZhQW5k0rKfhmfFDsf0+cOM9tU0mw8Hsx8vP9lg+LOkh7Br+9Z+DiMyg+5pNvMbLNC2hfwL8fr8XO4FX6uzyrIVT2W9XHvopm4V9AY4FNmdlwhvzPxr69WA9eQdH3xPi6RuQofe/Lt9OJcAn+JblQiWyVe1GkkE56ZvSs1qK4xs6IJr1F9bjKzDxbSVgSWJtfIBJa1kqB13WUomnfOx12xPoZ/Ou8L/F+J3AnAh81sLpD5A19Ah1tk5q2yDG6WuBe/CO/GL0Sni4U/iF+WNI8GLXMzOyr9fq7VQZjZE2odZbMdt0Rwhb8J3sE1KZk0it5K7dhRs1ZrXkmUmW3OwofJfzKtfwZ/+Ba1sNt9yeIjG2/EvZDAW1y/w1vi+eMZgXdqXysPObGepCXN7D9J4Q/Do7BOa1Z+qkP2wn0tKbjVyhR+kj0/KehtU9KuVh4aYRk8BlJ+VOwDwP6SJpnZYSl9lnyE6Bl4P9ICGgQLw78QtyOZeeSjc88xsz/nhdqwwX8DeG/2Qk0v3FvoOkK86bHgA41+mOr1JH79D8FfimUePk+mZam0NONueT/ehXTuy8t/ra1sZtPkHnIkE1MjV8uy/pt1CjJVTXjF+zs712XP72X4+KHpab93pTpUGk9RCevFXuHBsODulNDZw+aGErmyQFBlaVNxt8RsfUP8ASrKrVG2NKjj0rh75bfwFtR3SF4eOZlDcUV5dFrm4HE4enJu7ky/99DhPVEM9lQpAFib5bb0JqEjMNmteMf0LFy5/Qcf5l56nQtps8rk8GiJY3Hf/0uA8wsyN1Y8jpl4tMwVcWU0G/hpE/n34UrtYHLeQQWZ64ElcutLpLThwIMN9hkPvLtinScBf8Vb3Dfg5oxs29m55Qzc3NLFGwv3Jlkqt74U7rLZ1rHgfQy/wl1mM5nl8CB7J/bwHju7ZDmrIDMTD6WRBeHbjBLdkL+X6KxHbinI3J6OLcvvbZQEdyzc3zNwj5wz8P6KotxO6TqNwBugD+Adu732PA7Fln5ml39WHg/7GdycUWRW+nzMfJn3xh/iIu+0nCeLmd0vqYtvrnnLvEss8QZ1vJQOX/hGQb72x1sS/4JFnje34p+cpLQjzOzHahCW2Lr6AFeJvVM1AFg7ZptXJX3QzG5K+21BR0CrrK6T0rapwAHWEW9lQ7yFWGSGpL1wLyRw88kVZdU0s1ck7Q+cnM5XsdPw2mT7/x2dW4lFu/goM/tnMnecbf6l0Ch++ndwu+1F+Fff2ZIuNLMfFETH4g94ds5G4DbnNyS93uwLSNL7rGT0c2qJ74N/UT2Hv3Sm4+MGLsQHjGEtvjhz5sO/4gOjLsXvs10o/8poeiz41/e6lrRbqsM/JR2IjzM5LJXbdmjlVseS+Bp+HtaWdDOupPdoIFslXlTV2FOL7u9WmNkVkpbEXwxvxb8QG8Xz6RZDUen/ICmjw3EFuRxufy1yIO4lcQj+UN6I9/wXeUgeXfE8/AbchxKXOlWMJZ4YZ2Y7tjiOKlE2s3rMogJm9on097upk2wU8IeCWNUAYFDBbJM4EI88OSodwwv4SN4ymr5k1XlU59fwcwz+ybwAfwnlkVpHXc1s93mvmTJ323aiKk7GTSKvpUoch7uNFpX+j3HFMpOOjrsfJrPUH2nfzAjeOPgNrjCeTuUfZmYnyqOWktLWwj1eNkvHeyvwVetwZczMD39JS8allNPqWHbJK/yM9FLIp7c7up7UL1OW9+dz/++Sx+rJvLQaOm/g9/Jw3PPnq3if3+55AetswhONTXgtG0glDbfl8GfxYPko5LLAet1iyHTkykMRfBn3m54DnGktev3byPdAOi7WjcBp2cOck7uH5A9uHSMr77PyIeOn463Ohr7wqZW1L96SEN66OsfMTuzGMazYbHu+RauKI5WTbFsjWeVRJzGzfzap6wV4azv/kh1pZpObHUOT/LbCGwA3m9mPkqI7rDsPkaQ9cFPcTWb2lZTX8Wa2e4nsVcBkS+Mg0hfWeWb2sRLZMfiAOeFxbbqEeEhfQMcUv4DMbL8SWRWVq6QnzWz1QtpteMz2zP9+L9yE+H4akPpARja6hs2ORdLvcVfc4uQr+wCfLGvB52RazUyVvwbL4OMznileZ7Uxu10VVN3p4SK8gZT5638G2NjMdkvb921WjlX3829d5yGk9H+Hm3b+hPswP2ENwhAk+ZbxYNosP/OuuMu8c2cE7hlTpvQfxF9OTX3h1RFlE7wj9+7C9unN6pQ9ROrwES4dQFZ2zGoRACzJ3Ap8o2C2+YmZbZ7W2/IuSvtUfcl+qLhvyrPh1I+NFJakzzbIKx9cazhwiJn9rFH+hTx/j3eaX4uf++1xD6PnU96H5GRbTmHZ7gu2pD5PmdlqhbTbiwpe5d47v8UbVG/gJslReF/G8SXlNDwW+ZSUF+OmvWxQ4yZ4yJFPmNlfC3nNpDAzFW6Db3pfpX2H4f0O+VnhSuPpFK7FNDP7pLrG/yGtv4D3P1yqzk4P2Zd4l+c45dvroT66y1Ay76xvyfVKbqtvNU/rmZTEgynSxsuhaixx8JdSVYR70ZQp7M1xG/wF+Kd+6SQrZtZoooiuhVUPAAblZpt8i6Vd7yKScv9ZWpqR98leBm9dzqZrsLcuCktSUWHlXeyWwT/X7yI38CmZID5eoV4Zl6QlY2aZkCpMYZl4uIqZsQmLFFjuy2+GpCNxZwXDXUzL+kXWT7b3vfGBYd/Ez2Unpd/qWJJSf7+kbXCXauGeU9c1qHPlPpQS1sHDNuSZSIt4Oqn+4P0PZayMewheSnuDx1r2a6X0dfDxJMWR191qjJZivdgrPJALXafAu6uFfNV4MA/jSnoVvOd/JRp4t+CtueNxW+T2TfJcvWwpyHwHN1N9F3ervBf3887LDAd2xD8Z78btxRs0KXcL0hSMuNL4aUm5t+P2y3wsmKbxeHD7YzZh9949vI7r4EPpH8Rtmo/hQcta7bcacEFJejY15N7peJekxVSZeEt2ekn6Mbgr35a4Z877aOCV08bxtpzCMqUvgzdSspfJV/EY73mZ+cA/S5b5wMKc3OPpvD5esnQ517gHSRa+YKuUVjYtZqVjafPcjMFHKW+S0hpNR5gde/b7ZwrTZ1Ihnk7FemXTOc4g563UYp/3pGd4Hj7g8m5KPLDwr8Ft8Zhfa5Ce/57WuVMZvZnZQC6k4Ff5mzx/E+Tksof1OFxBb97sAabiy6Gwz8rQeL7OdDPfl34fSXV9oCDzUP6hxj+BH2qS59J45+j/0cC1k45JmTdO/w+l4LJGhQBguIKfgivA7VOe/5Vu6EtLyl0Xd/u7P62/m8ILrKc3fapDlyBgVFRYhX2WxEMxFNNnlCylge2o+PKimhvtcEpcJPvx2ToE9+C5Mp3nNXBzY9vH0ma5e6b74NS0vhY+yUt385sBvIgHZZyeLQWZ4ouz08skyXwtLWem+3VKLu1rLeqwHLBck+2Zy/mcXFqXc92TZciYd6z6zDLFEMGt4sE0DRYmH7l3HG7a+D7uebAyMEzSZ82s6B2DFUYAJtv9lwpi8/BWU2bLXprOHhTZvkvjvr2T8Q6qk+gafyZjoZmZpF3wmPBnlnQgVQkA9hv84bkVn0TiCNx3e1czu6ek3DNwc8wvAczsvmR2KXqygI8+vC51Rj6Bexr9iYJXTsHbIQtjXBy8RipzXtp2o3xUbdGmn3cNHIZ/WncZrGUV3e4SZ6c6/wz3lf8c5ea3lm601l489sqkfoqd6Nqx2amvxcxOwu+rjCfkg62KVHEJroz56OULc+uPUfCgyaOO+ECGK8rfF0S+W6HMKibJTKZs8Fip6UiFEeSSGo0gfy31Rzwi6b/wl+0qFepUmSHTkdsukpaxrp2DKxUvglrHd5+FD7IahQ8y+YiZ3SYf7XqBVYy1knUA59bzHYHgIyw7dQTKI/BtCFwFTDWz+1uUcQPuovk5vKP0//CW2EY5mZVxN77tcCV1DT4BRz72/Rzr6D8Zjk/kvLqZzW9Q7p1mtolyM0c16sSS+09vibeSr8dv+uPMbL2CXP5ltRCYZ2Y3Nzv+3L5LmI/G3MHMrlbnKfcW4p/fm1shZELadye6hvgom2BmtplNKJyrP5nZlkXZ3D5bUTKFZdpWOR57VSRdiTcqilMwfi9t38fMzmvUIV98OVQ9lgr1Kh13kiu3LAb9qbhzRD4S6F/M7KCC3Br4WJo/SnoLPt1lo/s2c6Qw3GOr6EixZ/EeKUtL6dfiTgmZi/HewNZWmE1L0iZ4I2t5vBG5HO4hdltZHbvDkGnpd4OLJO1iya1THlDqCgqzU1Vo3S1habJrSUdnF8d8dqfSHQoP0TDctFQMFXE1bhJ5Ezddlb18PoMrgHWBQ3LlNQpS1nCS8AzzCJx7NzxaZ5Fvc2qFPt7owUn8XdLasCjE7R74YJcyDsNH0B6C3/Tb0LlzOCv3XPkAOMysLMwGqayyaKPgo5yvlAdR28e6eo5MIdfKTGn/k+o2CR9ZugeNHQYqt9jU4fb3eEp6O96CzHMF5Z2sPWGcNQ/glw1GKmv9dlLKKky/aNXDNpdRadxJga2ADS21YlODqJNLdHKwOAAfUb02Ppjsf+gIlZGXzQbXZV/N56jr4Lou90iDNPDgat/Prf9AHuAuX+Zw3HX1G/iYkyoDztqnN21Fi9OCmyV+j9tLx+O2ww+XyI3COwBnpeUE3Ksg235X2f+y9Vz6Ubnl27iSXSZtWwIf5PJ33EPi7vT/eGDJPjwfR6Tfk/FP+U5LQbZS/0lOfi18cM4ruPK7CRjfzXoK/0z/Ox4j/kX8hfmdBvKH55Zv4yaps9K2u9N98CQeXz2/X5fh9HRMUpP9jsQDbJWVu0naPg439VwMbFYid3A6lgdwJTWHxp2Vy1IydL8H1/xHZfd8xX0PK0k7nwaTnPSwniMqyFxMLuwJ3u9wQUHmHtwMk++vKp0MhiZ9arhjx8m4q2b+OTmbBn2AuHPHXngjbxg+wK9LXxUpqF1vn8P8UtuWvpmdkWzWv8eV/pesfMLwVqNON5b0T1wZLZv+k9aXoQQrGeiU43i8ZbWmpdazfFDTT9K2w5rsW4qqzeP7pWRaadnKsur9J5n8Y8B28rELw6zkq0AVxxzgx78F7s3xeNp3LeA0SV+1gh+9FaZ5lE/SMr1js52RzF7nS/oocJD5hOhl5oXMHPiKpFXxl06pO6x1RFtt1WKr5PYnaWf8HlgKWFM+SvloazKgqQK3AZekVvp/aPyFWMbX8PDIecYAD8gnHmk4/WJVVMF9ONcfMwofPX9HWn8/HhQuz+tm9u/si1geZbORGWkejfvUnsEbZB+nc+iWNeiY4jQro90R5HcDl8qnQG0UOK5H1E7pF0wrWUjle4DNJG1mXe2Ua1vnEZffk4++BdpTgBUVW6X4JO1gKXyrNe+kOhlXKmPwGDQXWHmnbNtI+iE+2XM2OnUFfJq6fJySSmMO8GiH21tuIhgze0w+qvMaWvvRv4VCeAUz+3NSMD/AozV2Gqwl6TB8SsXpqaPyx7gfv+FmnrxsuzOzVY119F18LMLMlM89kiqPv2jACfh5n5O/3ypSdn2aNWa6w4nADqSXtJndq66D8tqZFOkGSd/CG2fbA1/Bo1ouItef8Dr+AisOrsM82u29ks7H+3c+jTcKH8djLS2ixTNXxop4YyLvUGI0ds5om9opfbraJy9pkJ5RaVBFRaooNit7AK1rfJK2kfQbM/tMWZp5eIcTU0fXXniAsGVSXS+wngV9+ojlJuYwD0H7UToHp3o7/mBNxh+iK1K5DxTyWtJKZv4ys/+TB6rqhDqPrByOB9nKOl4XnX/zvp0j5RPDX5DkMsbhndvvwju4bwG+gI+4LrbQMyW0WzqmrGU3GW89ZvXKGh9VYx0tNLOXC/1EPfXCeAR3o+1OPov2UR+FQAEws6cKx/xGYXs7fQdH4vGX5uC2/SvM7FcFmexLdzYNBtfJw7DvRcc0lL/DTTJN+/8kvZuunlIXp23jzOxpKwkcl77yeo3aKf0WppUyvgz8Wj7qFNyG3KVjsSJVFNuDclfPsvgkVad6a0SnCTfS522x4/oJ3Nb7I0nvxc1bR+EKs7sMl7S0mb2eyl0W/1zOl/sG7ln0B7kb6mRcGR5tZifnRJt5gpRty4+sXAg8l1NIXe4FM5spaQI5F1oz+3qq91K4i+8HcJPN6ZJeMrP1c7I3JNnvm1m+VXpZ6jTOaOb2V8b9kj6Nn8t18I7uRlM1VuVZ/BxfRckLp8QcmCHcxp1xLp1DoKxPx8jWntDO/MG74fftKql+i0xVchflcWZ2CnBG6tB9GzAhXb9sXl2sWoybh/Fj3dnSfAqSyoI65ut3Fj4+5QE6z1ORteCvk3uTzSvs9zm8cdTpi6RH9GWHwWBecNe35XPrK+CzKjWSXzSogpJOrG6UXzqYCvcouB1vWZyAtxxvwL1ExnazrCl07nDNOl3/ARxbkF0Sn+v0fHxKwN/h/vc9OdYj8E/j/fHwFDeROo5LzsluuPfDnficv2MLMvlO5OIgmv8UZIfRYjRxm8cxCh8B/X28Y3oWHh6gTPYhYK3c+po0GVxXoey34COC70zLD0gDoHqQ51FlSzfyyQ8kWoIWo+HbyDcLefAc7qp8Ho1Hwz8KvKvBtpvxYG3Z+j24GWV14LoG+zxOblAducF1eDC33+Ff7Wfg3j+PtziW0rkRcts/in95rZNLm4J/lYzrrXvYzGqt9Msm9iidAKFE7skelNtSsSW5bXDPjkOAbXvpmI9tsm17vFX/HN6q2JsKXhMVysz6TXbEX2AnADuUyJ2Lf1L/AHe9663r3GOPEnz8xc34l8j38NbsCi322RFvwc9My7wGx71uyv8a3HPjekpG+VLwLmqUNhBLUcn3ltJvsw43N9l2Z2H9F7n/tzXYZ6XcMhbvSzu6IDMiPSeX4x24p9HAGwrvkF6/xTFsi7+8NsT7M25udZ91Z6nz4KzZeGS/J9P6GsAllhsg1WTfLhELK5bZ1mCqvkANoiDKB6H9Fh/m3vak2i3KrDJp/Jt0eCs08jDqTtnX4+6T3fYoSXb+lXEvrltwt8+WtvBkpnpnWn3YknmrIHMv7iveKfCfmc0uyN1VvDfL0tpBFeeTrpDPG3Sc28z08wrdvHbq3uCsn+Pm09/T2VR1saRHzewdDcr6i5mtXbFeXea0zW1bEffr/1TZ+Usd0JfhX8/NIut+MB3DLbjP/mvFvHpK7Wz6Ob4N3JTc9cBHqB5Qcd/uvinbHUzVq6hJFERrL8RAu9wmaRNrMmm8mQ3ro7J77FFiZjvKL9YGuD3/cGBDSS/gnblFtzuKHkC4ay/WNXb7QjM7rVHZkj6Cf/qPVeeJx5fDzXU9oep80k2xNl14K5B3G/4eXd0ay1gOf9F8OJeW2cxvl/RFM+sU9VbSl2gwuE7V57T1gryh9Mu0lHEW/vx3Gv2cKy/v2rk03up/Pt13vaobatvSh0UhB7LY4bdZziukVSeWmS12L8zkyZJNjP4epYnRzexTfVzug/iL7gkaTBrfH6Tr/Y9WLfQWeYzDxwl8AFeWK5nZ8iVy+c7nReGazWyPgtx3cXv1JXRuob6Qtm+MxxU6Go+8mjEfmGFmL/bgWLJQEYsm+5F0g5lt1d08exvlQnf0II9V6PgCyKaXnIAr113N7LmSfWbkVhfi5rmfmNncbtbh+na/oPqKxU5x9TIfoGOyDnDbHNAt/9rFgdfM7DVJJG+ahyWt13q3HtPO/AG9groRCK9JXofg98oWuJfKzaSRvRSG+meY2cGFPEbRMQ1gnswTLD8/gJHGEliHT/hvLU3tp45ZpLqt8BNV55MeSFqZ0FrOE21mzwMfUEccf3B3zesbFtr7X74Py4MMXkbB/NTL5bSktkpfPmfpJvgnLsChkrYwsykDWK2+plejIFbF3A00a3GVjlLuA35BRyC86ykEwqPr3MDNGI8HgPuqmTWKGdSKV/C+lE5Y9QlurpVP4rJoFqnUKm85i1QTqs4nPZipPE90UvINFT3QKsic4Y2I6d144S6LK/sy81O/UlvzjnwGnveY2ZtpfTjuvdOvJoeBQj2IgtiNsj6Oe+2sipsy1sDdFzdoumPPylwUxVPSQ2b2rty2HpsMKpRfGq7ZzI4syL0FH56/upkdkHzw1zOzywtyd5vZe1O/zGqWZpEaivdrwbT6FjpCG/R535ekL5nZL9Vgrmjcm2dTK0wpuThR25Z+Ynn8zQ2uAIcs6t0oiO3yfbzv5I9JcU3CB1/1JfnOsuII6v5o6eTDAyzE52x+ukTubNxz5wNp/WncnffygtwS8knHP4k7IXSb7njH9CftmFbVftiLVmVncz50cQCQdJiZfU9Sl1DaFeq5Lu7SOdrMNpSPzv24dY7a2S/UWekfi8dZmYG3ID6EmwOGJGb2pqR7Ja2euan2I/8xs39IGiZpmJnNkPSjPi6z7UB4vUn+pZp1IDcQXdvMPiVpctrvVak0JvfReLjtm8zsTnmQue6GxuiOd8xgpVLYi17ia/ik6I3CdTejnYmE+pTaKn0zu0DSTNyuL+CbZva3ga1Vn9OrURDb4CVJI/FJJM6X9Dw9dzdsSh+4EVaiGx3I/5aHpbC0/9rkOvoyrM1ZpJphuVADqfVaJfTAoMSqh73oDRoFAazCW8zsjsL7vE+fgUbUVulLus7MtqUjzG4+bajS21EQq7ILbmL5Kj6CcRQdQc+GGu12IB+V0laTR23cAg/P0Ql5RM2D6Rqwq6cv7KHSqfc2SWull2F2vt7WYp926cm5amcioT6ldkpfHhHwLcDKyfUte/Uuh3c0DlnKTA498Vlvo9zsq+JNeTTJfil3gGhrJjUzu1bSXXifh/CpKbtEEcU9rs7EXf66DO4J+CoePO6xtD6ervNOt6TV+Jxu1w4OwsNtvFPSX/HYPvv0IL9uUzulj98Ih+EKPj/UfT5wyoDUqI/pTZ/1xaHcAaY7Hchb0TEX65J0Dumb8Zr5BOU9pugdU+jv6FPvmL7CzP6QPJ+ahr2okE9fjc95xsw6TSSUGl79Tu1cNuUTDz8N7GFmJ8sn2N4d7/T5rvVy3JnBgHpp8vbFpdyBRB1xaPIxaEjry5jZkgX5qhN6fxr387+GzoN77qLGZIOz0v9Ok5JL+qHl5nEYSOSj4b+YfflJ2h0PgLhuv9elhkr/LmA7M3tBHgRpKm4rfQ8emnWPphkshgyUz/pA+8ovDkh6gM4Teg/DQxUX5z44Fo/d8hdy8dhtkAztHyiUCzqnQgC64vpAImkjfAT3TNzKsBLwhQZuvH1KHc07w3Ot+U8Bp5vZRcBFyk2DOMQYKJ/1gfaVXxyYi8d1fyKtrwbcVyL3CTw+f58OpFsMUYP/ZesDhpnNkXQMbuKcD3xoIBQ+1FTpS1rCfPakbekcWXOono+B8lkfUF/5xYSV6JjQG9yF+Fal+ZRz3jn34oMJn+//Kg5qrMH/svUBQ9KZwNr47Fnr4i6lvzCfzatfGapKrhkX4BMk/x1vff4JQNI7qDZB9WLHQPmsD1S5ixlVB/qMxoN23UmHTd/MbJe+qdZiw+LSsLgfN+cY8HhycijOg9wv1M6mD4u8SsYA12TuhGmY9Mi6d4wFdnjJhAAAAdhJREFU/Y+k0XgLH+AO86iQRZl8uGPh3j6T+zJ+UTA0qaXSD4LBgqRPAsfjHXwCtgS+YbnJunOy7wE+jcffeRy42DpPGh8MUiQ9Tnno57X6uy51NO8EwWDi28AmWetePoXhH/FQztkX6F54LJl/4DNdyfp2prOg95mY+78MPrXiigNRkWjpB8EAImmOmW2UWx8G3JulyecO/hOwv5k9mtIeG4gWYtC7qMmcu31JtPSDYGD5g6Sr6Tw468rc9t3xlv4M+QTtUxlErohBNdTmnLt9Wpdo6QdB/5O8xUab2c2SdsM7ZgW8CJxvZn8pyI8AdsXNPNsA5wKXZHF+gsGNus65+zg+5+6f+70uofSDoP+RdDnwLTO7r5A+ETjKzHZusu+KuE34U3Ufkbs4k8Jan9jv5YbSD4L+R9L9lmYxK9nWyc4fDE0kPWlmq/d3ucP6u8AgCIDmA4d6EsI3WHwYkL6ZUPpBMDDcKemLxURJ+9M55HcwdBkQM0uYd4JgAEijcC8B/k2Hkp8ILAV8ogZTd9aCVpOymFm/e1CG0g+CAUTSJCCz7T9gZtcPZH2CoU8o/SAIghoRNv0gCIIaEUo/CIKgRoTSD4IgqBGh9IMgCGpEKP0gCIIa8f8Bd2KLOcTvo1oAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.city.hist(xrot= 90, bins=31)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Let's make a feature for cities: 1 for capital city, 0 for non-capital city"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "capitals = {'London': 1, 'Paris': 1, 'Madrid': 1, 'Barcelona': 0, \n",
    "           'Berlin': 1, 'Milan': 0, 'Rome': 1, 'Prague': 1, \n",
    "           'Lisbon': 1, 'Vienna': 1, 'Amsterdam': 1, 'Brussels': 1, \n",
    "           'Hamburg': 0, 'Munich': 0, 'Lyon': 0, 'Stockholm': 1, \n",
    "           'Budapest': 1, 'Warsaw': 1, 'Dublin': 1, 'Copenhagen': 1, \n",
    "           'Athens': 1, 'Edinburgh': 1, 'Zurich': 1, 'Oporto': 0, \n",
    "           'Geneva': 1, 'Krakow': 1, 'Oslo': 1, 'Helsinki': 1, \n",
    "           'Bratislava': 1, 'Luxembourg': 1, 'Ljubljana': 1}\n",
    "df['capitals'] = df.city.apply(lambda x: capitals[x])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Also let's make a feature with the qty of restaurants into the paticular city:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "res_qty = {\n",
    "    'Paris': 4897,\n",
    "    'Stockholm': 820,\n",
    "    'London': 5757,\n",
    "    'Berlin': 2155, \n",
    "    'Munich': 893,\n",
    "    'Oporto': 513, \n",
    "    'Milan': 2133,\n",
    "    'Bratislava': 301,\n",
    "    'Vienna': 1166, \n",
    "    'Rome': 2078,\n",
    "    'Barcelona': 2734,\n",
    "    'Madrid': 3108,\n",
    "    'Dublin': 673,\n",
    "    'Brussels': 1060,\n",
    "    'Zurich': 538,\n",
    "    'Warsaw': 727,\n",
    "    'Budapest': 816, \n",
    "    'Copenhagen': 659,\n",
    "    'Amsterdam': 1086,\n",
    "    'Lyon': 892,\n",
    "    'Hamburg': 949, \n",
    "    'Lisbon': 1300,\n",
    "    'Prague': 1443,\n",
    "    'Oslo': 385, \n",
    "    'Helsinki': 376,\n",
    "    'Edinburgh': 596,\n",
    "    'Geneva': 481,\n",
    "    'Ljubljana': 183,\n",
    "    'Athens': 628,\n",
    "    'Luxembourg': 210,\n",
    "    'Krakow': 443       \n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['rest_qty'] = df['city'].map(res_qty)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### And another feature with population of the cities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "city_population = {\n",
    "    'London': 8173900,\n",
    "    'Paris': 2240621,\n",
    "    'Madrid': 3155360,\n",
    "    'Barcelona': 1593075,\n",
    "    'Berlin': 3326002,\n",
    "    'Milan': 1331586,\n",
    "    'Rome': 2870493,\n",
    "    'Prague': 1272690,\n",
    "    'Lisbon': 547733,\n",
    "    'Vienna': 1765649,\n",
    "    'Amsterdam': 825080,\n",
    "    'Brussels': 144784,\n",
    "    'Hamburg': 1718187,\n",
    "    'Munich': 1364920,\n",
    "    'Lyon': 496343,\n",
    "    'Stockholm': 1981263,\n",
    "    'Budapest': 1744665,\n",
    "    'Warsaw': 1720398,\n",
    "    'Dublin': 506211 ,\n",
    "    'Copenhagen': 1246611,\n",
    "    'Athens': 3168846,\n",
    "    'Edinburgh': 476100,\n",
    "    'Zurich': 402275,\n",
    "    'Oporto': 221800,\n",
    "    'Geneva': 196150,\n",
    "    'Krakow': 756183,\n",
    "    'Oslo': 673469,\n",
    "    'Helsinki': 574579,\n",
    "    'Bratislava': 413192,\n",
    "    'Luxembourg': 576249,\n",
    "    'Ljubljana': 277554\n",
    "}\n",
    "df['city_population'] = df['city'].map(city_population)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Another feature: Qty of people per restaurant into the city"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['rest_per_person'] = df['city_population']/df['rest_qty']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Reviews\n",
    "#### There is no missed values, i'm going to make a feature with qty of words into the each review:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['num_reviews'] = df.reviews.str.findall(r\"'(\\w.*?\\w)'\")\n",
    "df.num_reviews = df.num_reviews.apply(lambda x: len(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Price range\n",
    "#### Filling missed values with 0 \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.price_range.fillna('0', inplace = True)\n",
    "df['price_range'] = df['price_range'].replace('$','Low')\n",
    "df['price_range'] = df['price_range'].replace('$$ - $$$','Medium')\n",
    "df['price_range'] = df['price_range'].replace('$$$$','High')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Medium    18402\n",
       "0         13879\n",
       "Low        6276\n",
       "High       1423\n",
       "Name: price_range, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.price_range.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Cuisine style\n",
    "#### Filling missed values with 'not defined'  and cleaning the data into the column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['cuisine_style']=df['cuisine_style'].str.replace('[', '').str.replace(']', '').replace(\"'\", '')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.cuisine_style.fillna('not_defined', inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Making a new feature with qty of cuisines into one restaurant"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['qty_of_cuisines'] = df.cuisine_style.apply(lambda x:len(x.split(', ')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1     16539\n",
       "2      6292\n",
       "3      5304\n",
       "4      4793\n",
       "5      3605\n",
       "6      2042\n",
       "7      1022\n",
       "8       283\n",
       "9        76\n",
       "10       19\n",
       "11        3\n",
       "21        1\n",
       "13        1\n",
       "Name: qty_of_cuisines, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.qty_of_cuisines.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Number of reviews\n",
    "#### Treating NaN as 0, so filling it accordingly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.number_of_reviews.fillna(0, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Majority of the restaurants have less then 3 cuisines "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Converting categorical variables to dummy ones:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "city_dummies = pd.get_dummies(df['city'], prefix='city')\n",
    "df = df.join(city_dummies, how='right')\n",
    "\n",
    "price_dummies = pd.get_dummies(df.price_range , prefix = 'price_range')\n",
    "df = df.join(price_dummies, how='right')\n",
    "\n",
    "cuisine_dummies = df.cuisine_style.str.get_dummies(sep=',')\n",
    "df = df.join(cuisine_dummies, how='right')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Dropping the columns with non-numeric values:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['city', 'cuisine_style', 'reviews', 'url_ta', 'id_ta', 'price_range'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### I've heard it can be helpful to normalize some numeric values, so let's try:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.ranking = preprocessing.normalize(df.ranking.values.reshape(1, -1))[0]\n",
    "df.number_of_reviews = preprocessing.normalize(df.number_of_reviews.values.reshape(1, -1))[0]\n",
    "df.num_reviews = preprocessing.normalize(df.num_reviews.values.reshape(1, -1))[0]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(['rating'], axis = 1)\n",
    "y = df['rating']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating a model\n",
    "regr = RandomForestRegressor(n_estimators=100)\n",
    "\n",
    "# training the model on test data\n",
    "regr.fit(X_train, y_train)\n",
    "\n",
    "# Using trained model for rating prediction on the test data\n",
    "# Predicted values are into y_pred\n",
    "\n",
    "y_pred = regr.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE: 0.2059869934967484\n"
     ]
    }
   ],
   "source": [
    "print('MAE:', metrics.mean_absolute_error(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
