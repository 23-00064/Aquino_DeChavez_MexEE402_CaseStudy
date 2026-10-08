# Aquino_DeChavez_MexEE402_CaseStudy

# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Aquino, Kimberly Chezka R. |23-00272 |MEXE-4101 |
| De Chavez, Zoe |23-00064 |MEXE-4101 |

## Notebook links

| Chapter | Aquino | De Chavez |
|---|---|---|
| Ch1_2_3 | [link](https://colab.research.google.com/drive/1uzwUUM62UsJT8u53Ne9xzbici6U1P-Gm?usp=drive_link) | [link](https://colab.research.google.com/drive/1FgNZy6bVj1NN76ktEShaxNadtXu6Cz4_?usp=sharing) |
| Ch4 | [link](https://colab.research.google.com/drive/1Aq8-3vytmvoFdzw_g865FWNDF6uLZ5w5?usp=sharing) | [link](https://colab.research.google.com/drive/1Zk1lKVk0PpMEQy6ZLvN40HTDJaO1N-2Y?usp=sharing) |
| Ch4 | [link](https://colab.research.google.com/drive/1Aq8-3vytmvoFdzw_g865FWNDF6uLZ5w5?usp=sharing) | [link](https://colab.research.google.com/drive/1M4f6xoY6shIuioujIAs3HADHd-B6rb5i?usp=sharing) |
| Ch5 | [link](https://colab.research.google.com/drive/1Ff4T9rIZpmsCQegWKgplzqIM7Ix0MDIw?usp=sharing) | [link](https://colab.research.google.com/drive/1M4f6xoY6shIuioujIAs3HADHd-B6rb5i?usp=sharing) |
| Ch6 | [link](https://colab.research.google.com/drive/11iCOwjseur9i5FseJJ9UB9pJKbIhK7nx?usp=sharing) | [link](https://colab.research.google.com/drive/1X3sxYkThQUO8iO4HaB8z4GFqK0CbB_ox?usp=sharing) |
| Ch7 | [link](https://colab.research.google.com/drive/1DabI0ZFXgUoQWVphmmwxo2qgsW9FmoPM?usp=sharing) | [link](https://colab.research.google.com/drive/16AXlQmy-h2lyyDIqjQSNZyd8ce5ytkry?usp=sharing) |
| Ch8 | [link](https://colab.research.google.com/drive/16t8ceuUn077M8EEk2sPcwXTYhKer5R8K?usp=sharing) | [link](https://colab.research.google.com/drive/1dSe1AJNZOFt7raA5CMG6C6BVBQUAyU00?usp=sharing) |
| Ch9 | [link](https://colab.research.google.com/drive/10FP5yDqVVT40yUxGyFSOTJdjOMZDuaJu?usp=sharing) | [link](https://colab.research.google.com/drive/11aGISi8aN7-tAaR_KYkWAjdSbTi90oOF?usp=sharing) |

## What we learned

### Aquino 
### Data Preprocessing
### Learning Reflections | Chapters 1–9

**📖 Chapter 1: Introduction**

The data can have missing values, mixed-up units like mph and kph, or columns that don't help at all. I learned three ways to fix this: fill the gaps (like using the average), convert units so everything matches, and remove columns that aren't useful. It isn't only about neatness. Clean data gives a more accurate model and also saves time and money later. In the code, I saw that even the order of the cleaning steps matters, because filling Publisher first left nothing for the next step to remove.

---

**💻 Chapter 2: Learning Experience**

Before this, I thought you just give data to the computer and it learns. Now I know you have to look at the data first. head() shows a few rows, info() shows the column types and what's missing, and describe() gives the numbers like the average and the biggest value. 

---
**💻 Chapter 3: Learning Experience**

I learned that missing values can be filled in or the rows can be dropped. What surprised me was that my order of steps mattered. I filled the empty Publisher cells first, so the next step that removes empty Publisher rows had nothing left to remove. I also didn't expect that filling Year with the average gives something like 2006.4, which isn't a real year. I was also surprised that the Rank column was dropped, because it looks like data but is really just a number for the row's position. 

---

**💻 Chapter 4: Feature Engineering and Encoding**

I learned that I can make new columns from the ones I already have. For example, dividing lemonade sold by temperature gave a new column, and the numbers kept going up as it got hotter. I also learned about binning, which means putting numbers into groups like cool, warm, hot, and very hot. What surprised me was that no row ended up in "very hot", so that label was empty. The biggest surprise was encoding. Computers need numbers instead of words, but if I turn Sunny, Cloudy, and Rainy into 0, 1, and 2, the computer thinks Rainy is "bigger" than Sunny. That's why weather uses one-hot encoding, while Little, Medium, and Lots can use ordinal encoding because they really do have an order. 

---

**📊 Chapter 5: Scaling and Normalization**

I learned that the computer doesn't know what units are. Study hours (8 to 15) and grades (76 to 92) are just numbers to it, so the bigger numbers can end up controlling the result. Scaling fixes this by putting the columns on the same size. StandardScaler makes the average 0, and MinMaxScaler makes everything go from 0 to 1. What surprised me was that scaling isn't always needed. 

---

**📊 Chapter 6: Outlier Detection**

I learned that an outlier is a value that is very different from the others, like the 100 in my list of small numbers. I was surprised that the Z-score method almost missed it, because its score was only about 2.5 and the cutoff is 3. The 100 made the spread so big that it hid itself. The IQR method found it right away, so now I know one method can fail where another works. I also learned that finding an outlier isn't the end. I still have to decide whether to remove it or limit it.

---

**📊 Chapter 7: Feature selection**

I learned that more columns isn't always better, and that I can pick only the useful ones. There are three ways: the filter method (check each column's correlation with the grade), RFECV (remove the weakest column step by step), and LassoCV (push the unimportant columns to zero). What surprised me was that the filter method couldn't tell that "study hours" and "assignments completed" are exactly the same column, so it kept both. It also showed final grade in the list, because the answer is perfectly related to itself. So I learned I need to read the results and not just trust them. I also got a NameError because I forgot to import SVR and RFECV.

---

**📊 Chapter 8:  Constructing a Preprocessing Pipeline**

The data goes in, passes through the steps (fill missing values, then scale), and comes out clean, always in the same order. This is useful because my code is shorter and neater and I don't forget a step. What surprised me was ColumnTransformer. It only worked on Age and Fare and quietly left out all the other columns. I also made a small mistake by typing x instead of X, and the whole cell broke. In coding, even one letter matters.

---

**📊 Chapter 9: Full pipeline and Visualization**

This chapter put everything together. Number columns got the median and scaling, and word columns got a "missing" label and one-hot encoding. I also turned ages into Child, Adult, and Elderly, then made charts to see who survived. What I learned is that charts are not just for decoration. They help me check my work. What surprised me was that my own mistakes showed up here. One "after" chart used column 2, which turned out to be a one-hot column and not Age. Also, once I turned Age into groups, the later charts that needed real ages stopped working properly. I learned to keep a copy of the original column and to run the cells in the right order.

---
### De Chavez, Zoe 
### Data Preprocessing
### Learning Reflections | Chapters 1–9

**📖 Chapter 1: Introduction**

For chapter 1, there was no programming involved since it was only an introduction. I just read through the chapter and focused on understanding the concept since I initially thought that the information presented there would be directly used in the programming for the succeeding chapters. However, the chapter mainly introduced the terminologies and definitions necessary for better understanding data pre-processing and its importance.

---

**💻 Chapter 2: Learning Experience**

For chapter 2, I initially encountered some difficulty when downloading the CSV file. When I clicked the provided link, I was redirected to a website with a download option. I thought that this was the file to be downloaded but clicking the option shows a code instead. I then reviewed the page further and found the right file. However, the file I downloaded was in ZIP format, and I was not aware that I needed to extract the required file before uploading it to the notebook. Therefore, when I attempted to run step 3, the code did not work. I consulted Gemini, the AI tool available in google colab to determine the source of error, and it explained that the code was not designed to accept a ZIP file. I consulted this problem to my partner who explained that I need to extract the file first. After extracting the file and uploading the correct file, I ran the code again and it worked successfully. After resolving this issue I did not encounter any further difficulties in chapter 2.

---
**💻 Chapter 3: Learning Experience**

For Chapter 3, I already had some basic knowledge and a better understanding of how google colab worked. This gave me some confidence in completing the activity, as I understood that I just needed to paste the appropriate code and run the program correctly. I also made sure to verify the results were consistent with the instructions and code provided.

---

**💻 Chapter 4: Feature Engineering and Encoding**

For Chapter 4, I did not encounter any major challenges that made things difficult for me when it comes to running the code. I just made sure that the output that was generated was correct. However, when I clicked Run All, a “next step” option appeared, which I think was because of the AI tool in Google Colab. I clicked it at first because I thought it was a part of the activity, but I later realized that it was only an AI tool feature. Aside from that,I did not encounter any other problems or surprises in Chapter 4.

---

**📊 Chapter 5: Scaling and Normalization**

The lesson that Chapter 5 taught me was that scaling is an important step because differences in numbers could affect the way the machine learning model analyzes data. This is because the higher the value of a feature, the more likely it will have an impact on the data despite not being important. What I found surprising was that scaling may not always be necessary depending on the machine learning algorithm.

---

**📊 Chapter 6: Outlier Detection**

Lesson in Chapter 6: Outliers can influence the way we interpret data; therefore, we must be able to recognize them prior to coming up with any conclusion. Surprisingly enough, using various methods yields different outcomes since the Z-score method does not recognize 100 as an outlier while the IQR method recognizes 100 as an outlier. Moreover, not all outliers need to be discarded as some of them contain valuable information.

---

**📊 Chapter 7: Feature selection**

Chapter 7 helped me realize that all the features available in a dataset are not necessarily relevant to prediction tasks. It has been realized that using proper feature selection may lead to improvement of the performance of a model as well as reduce unnecessary features. What intrigued me is that various feature selection techniques can select different features using the same dataset.

---

**📊 Chapter 8:  Constructing a Preprocessing Pipeline**

In chapter 8, I learnt that data preparation does not only involve cleaning but also ensures that every task is performed in the correct sequence. It was clear that arranging the tasks involved in the data preprocessing into one process will ensure that there is consistency in the process, and the chances of making errors are greatly reduced. The surprising thing to me was that different processes could be merged together instead of being done independently.

---

**📊 Chapter 9: Full pipeline and Visualization**

From Chapter 9, I have learned that data preprocessing is vital as it makes the data organized, coherent and easy to work with. Handling the missing values, categorizing data and producing visualizations may help us discover the patterns present in a data set. What surprised me in Chapter 9 was the fact that the Titanic data set may reveal some relationships among such factors as age, gender, passenger class and whether or not a person survived.

---
## Errors we found

### Aquino
**📋 Overview**
> Ran the Chapter 9 notebook and nothing crashed, but a few charts and results looked wrong. All of the problems come from how Age was handled.

**🔍 Identified Issues**

**`01` — Discretization Order**

The discretization happens after the preprocessor was already fitted. The pipeline scaled the original numeric Age, so the binned Age only affects the plots, not the model input.

First, the preprocessor runs on the data. It scales Age and Fare and one-hot encodes the other columns. This result is saved as titanic_preprocessed.
Later, the code turns Age into Child, Adult, and Elderly using pd.cut.

**`02` — Histogram Visualization**
**Affected Cells:** `29–30`

titanic_preprocessed[:,2] was used as the "after discretization" data. After preprocessing, the columns are in this order: Age, Fare, then the one-hot columns. So, column 2 is the Embarked_C flag, and the histogram shows just two bars (0 and 1) instead of ages. The "before" histogram also runs after pd.cut, when Age is already words, so it can't draw the original ages.

**`03` — Histogram with KDE**
**Affected Cell:** `34`

Once Age became Child, Adult, and Elderly, histplot with kde=True no longer made sense. A smooth density curve needs numbers. A count plot works better for age groups, or the plot can use a numeric Age column.

**`04` — Correlation Heatmap**
**Affected Cell:** `39`

The heatmap only takes numeric columns. Since Age was replaced by labels, it was dropped, and I couldn't see how age relates to survival.


### 📊 Issue Summary

| Cell Reference | Visualization | Identified Issue |
|:---:|---|---|
| `26` | Binning | Done after the preprocessor was fitted |
| `29–30` | Before/after histogram | Plots a one-hot column; "before" runs after binning |
| `34` | Age histogram with KDE | Age is words, not numbers |
| `39` | Correlation Heatmap | Age is missing |


---
### De Chavez
**📋 Overview**
> No syntax problems were detected in Chapter 9, but several problems were noted regarding data visualization.

**🔍 Identified Issues**

**`01` — Histogram Visualization**
**Affected Cells:** `29–30`

Specifically, the histogram in Cells 29–30 is based on a wrong column from the preprocessed dataset that may influence the results' accuracy.

**`02` — Histogram with KDE**
**Affected Cell:** `34`

The histogram with KDE in Cell 34 is built on the Age column that is represented in categories. Actually, the count plot should have been used there because it is suitable for the representation of categories and survival outcomes.

**`03` — Correlation Heatmap**
**Affected Cell:** `39`

The Age column was not included into the correlation heatmap in Cell 39 due to the fact that it was turned into categories. This problem can be overcome if the numeric version of the Age column was used instead.


### 📊 Issue Summary

| Cell Reference | Visualization | Identified Issue |
|:---:|---|---|
| `29–30` | Histogram | Wrong column selected |
| `34` | Histogram with KDE | Categorical Age data |
| `39` | Correlation Heatmap | Age column excluded |


---
## Note on AI tools

### Aquino

**📋 Overview**

> Claude AI was used to assist in identifying errors in each chapter. It also helped explain functions and syntax that were unfamiliar.

**🔍 AI Assistance & Error Identification**

**`01` — Debugging a Code Error**

While entering my code, I ran into an error and asked Claude AI to help debug it. After reviewing the code, the cause turned out to be a simple capitalization mistake: I had typed a lowercase x* instead of an uppercase X*. Because Python is case-sensitive, the two are treated as different variables, which caused the error. After correcting it, the code ran properly.


**`02` — Understanding Unfamiliar Functions**

I asked Claude AI to explain functions that I was not familiar with. It explained what each function does and what happens to the data when the function is applied. This helped me understand the purpose of each step in my code instead of just copying it, and I could then apply the functions correctly in my work. This also helped me in answering some questions.

## 📊 AI Usage Summary

| Category | Claude AI Usage |
|:---|:---|
| `01` Debugging a Code Error | Error identification and correction |
| `02` Understanding Unfamiliar Functions | Understanding |

**📝 Scope of AI Usage**

Besides error identification and understanding, Claude AI was not used for any other purpose.

---
### De Chavez

**📋 Overview**

> Gemini AI was used by me mainly to recognize the errors that arise during the code execution and make sense of the error.

**🔍 AI Assistance & Error Identification**

**`01` — File Upload Error**
**Affected Chapter:** `02`

For example, in Chapter 2, there was an error due to the uploading of a ZIP file instead of a CSV file which cannot be read by the program. The Gemini explained what kind of error it was and recommended changing the code in such a way to make it readable for ZIP files. Yet, I preferred downloading the appropriate CSV file instead of changing the initial code.


**`02` — Undefined Variables & Functions**
**Category:** `Code Execution Errors`

Moreover, Gemini helped me recognize the errors arising due to undefined variables and functions which I had not executed prior to that. This allowed me to understand the reason for these errors and ways to fix them.

## 📊 AI Usage Summary

| Category | Gemini AI Usage |
|:---|:---|
| `01` File Upload | Error identification and explanation |
| `02` Undefined Variables | Error identification and understanding |
| `03` Undefined Functions | Error identification and understanding |

**📝 Scope of AI Usage**

Besides error identification and understanding, Gemini was not used for any other purpose.

---
## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.
