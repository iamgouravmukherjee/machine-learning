## Important concepts about Regression 🤔

#### **Multi-collinearity**:

> **Multicollinearity** refers to the phenomenon when two or more features (independent variables) have a strong correlation between them.

**Example: Predicting House Prices**

Suppose you are trying to build a regression model to predict house prices based on various features of the houses. Two of the features you consider are:

- **Square Footage (Size)**: The size of the house in square feet.
- **Number of Bedrooms**: The number of bedrooms in the house.

**Impacts on regression analysis**

1.  **Challenging to determine true relationship**: It makes it harder to determine the true nature between certain independent features and the target variable.

- In the above example, if the model shows that increase in the price is linked to both **size** and **number of bedrooms** it becomes unclear whether the increase in price is due to the **size** or **number of bedrooms** or **both**.

2.  **Difficulty in predicting practical significance**:

- Multicollinearity can make it difficult to figure out the actual impact of each feature on the price.
- For instance, it will not be possible to say how much an increase in the number of bedroom impacts on the price as its effect is linked to the effect of size of the house.

🔎 **How to detect multicollinearity:**

1. Checkout the correlation heatmap of the independent features. If there is a strong correlation (towards -1 or +1), then it hints the presence of multicollinearity between the features

❌ **Handling mullticollinearity**

1. **Removing a feature:**

- If two features are having multicollinearity we can remove one the features from the model. For instance, we can remove either the size or the number of bedrooms.

2. **Combining the features:**

- We can combine the two features and create a new feature for the model. For instace, we can combine the **size** and **number** of bedrooms and make a new feature called **size per bedroom**

3. **Regularization techniques**:

- Techniques like Ridge Regression or Lasso Regression can help by penalizing the magnitude of coefficients, effectively reducing multicollinearity.

#### **Regularization Techniques**

1. **L1 Regularization(Lasso)**: Least absolute shrinkage and selection operator

`Loss_function_with_L1 = Loss_function + λ * Σ(coefficients)`

  - Imagine you have a dataset of houses with various sizes and numbers of
    bedrooms, and you want to predict their prices.
  - When you use Lasso regression with an appropriate value of alpha, it may perform feature selection by driving some coefficients to exactly zero.
  - In this context, let's say you've chosen an alpha value that's suitable for your dataset. Lasso examines the relationship between house prices, size, and the number of bedrooms.
  - If it determines that "Number of Bedrooms" doesn't significantly contribute to predicting house prices, it might set its coefficient to exactly zero.
  - Effectively, this means that the Lasso model removes "Number of Bedrooms" from the equation, treating it as if it doesn't play a role in predicting house prices.

2. **L2 Regularization(Ridge)**:

`Loss_function_with_L2 = Loss_function + λ * Σ(coefficients^2)`

  - In contrast, Ridge regression applies L2 regularization to the same problem.
  - It also examines the relationship between house prices, size, and the number of bedrooms.
  - However, instead of driving coefficients to zero, Ridge encourages them to be small but not exactly zero.
  - So, in the case of "Number of Bedrooms," the Ridge model would shrink its coefficient, making it smaller in magnitude but still nonzero.
  - Ridge doesn't remove features; it retains all the features and reduces the impact of multicollinearity between "Size" and "Number of Bedrooms" by making both coefficients smaller.

##### How can I determine the right alpha value for regularization?

```
  from sklearn.model_selection import GridSearchCV
  from sklearn.linear_model import Lasso, Ridge

  # Define a range of alpha values
  alphas = [0.01, 0.1, 1.0, 10.0]

  # Create Lasso or Ridge regression models
  lasso_model = Lasso()
  ridge_model = Ridge()

  # Perform grid search for the best alpha value
  lasso_grid_search = GridSearchCV(lasso_model, {"alpha": alphas}, cv=5)
  lasso_grid_search.fit(X, y)
  best_alpha_lasso = lasso_grid_search.best_params_["alpha"]

  ridge_grid_search = GridSearchCV(ridge_model, {"alpha": alphas}, cv=5)
  ridge_grid_search.fit(X, y)
  best_alpha_ridge = ridge_grid_search.best_params_["alpha"]
```
#### **Dummy Variable Trap**
The dummy variable trap is a common issue that can arise when using dummy variables to encode categorical variables in regression analysis. It occurs when you include all levels or categories of a categorical variable as separate dummy variables in your model. This can lead to multicollinearity, where the independent variables are highly correlated with each other, which can cause problems in regression analysis.

In the context of predicting housing prices based on "Square Footage" (Size) and "Number of Bedrooms," let's consider a scenario where you want to include the "Type of Heating" as a categorical variable:

1. **Original Categorical Variable:**
   - "Type of Heating" has three categories: Gas, Electric, and Oil.

2. **Using Dummy Variables Without Trap:**
   - Without considering the dummy variable trap, you might create three dummy variables: "Gas," "Electric," and "Oil."
   - These dummy variables take binary values (1 if the condition is met, 0 if not) to represent the presence or absence of each heating type.

   | House | Size (Sq. Ft.) | Bedrooms | Gas | Electric | Oil | Price ($) |
   |-------|----------------|----------|-----|----------|-----|-----------|
   | A     | 1500           | 2        | 1   | 0        | 0   | 200,000   |
   | B     | 1800           | 3        | 0   | 1        | 0   | 220,000   |
   | C     | 1600           | 2        | 0   | 0        | 1   | 210,000   |

However, this approach can lead to the dummy variable trap because the presence of one heating type can be predicted from the values of the other two dummy variables. In this case:

- If you know that "Gas" and "Electric" are both 0, it's evident that the heating type is "Oil" (since there are only three options).

This perfect multicollinearity can cause issues in regression analysis because it becomes impossible to distinguish the individual effects of "Gas," "Electric," and "Oil" on house prices.

##### Now, why is this a problem?

Multicollinearity occurs when two or more variables in a regression model are highly related or correlated with each other. It's like having two or more "predictors" that are giving almost the same information, making it hard for the model to understand which one is causing what.

Now, let's use the example of "Type of Heating" (Gas, Electric, and Oil) in the context of predicting house prices:

1. Imagine you include dummy variables for all three heating types: "Gas," "Electric," and "Oil."
2. These dummy variables are like flags. They are set to 1 if a house has that type of heating and 0 if it doesn't.
3. If you know that "Gas" and "Electric" are both 0 for a house, you can immediately figure out that it must have "Oil" heating. Why? Because there are only three options: Gas, Electric, and Oil.

So, in this case, the presence or absence of one type of heating (let's say, "Oil") can be predicted just by looking at the values of the other two dummy variables (Gas and Electric). This is what we mean by "one variable can be predicted from the others."


When the regression model sees that these dummy variables are strongly related (like brothers or sisters who look very similar), it gets confused. It can't tell which one is really causing changes in house prices because they all seem to be pointing in the same direction. This confusion can lead to unreliable estimates of how each heating type affects house prices, making the model less trustworthy.
