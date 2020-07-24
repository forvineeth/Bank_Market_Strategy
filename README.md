# Bank_Market_Strategy

We have a csv file of banking data and based on the avaiable data, we need to check whether the martketing strategy is succesfull or not. I have used logit classification algorithm to check the efficiency of the bank's marketing strategy.

# within the csv
Note that interest rate indicates the 3-month interest rate between banks and duration indicates the time since the last contact was made with a given consumer. The previous variable shows whether the last marketing campaign was successful with this customer. The march and may are Boolean variables that account for when the call was made to the specific customer and credit shows if the customer has enough credit to avoid defaulting.

We want to know whether the bank marketing strategy was successful, so we need to transform the outcome variable into Boolean values in order to run regressions.
