library(tidyverse)

library(broom)

df <- read_csv("USstocks_balanced.csv")

#Portfolio Formation

#testing period 1 1992-1995
pf_groups <- df %>% filter(between(year, 1980, 1986)) %>% 
  group_by(permno) %>% 
  nest() %>% 
  mutate(betaest = map(data, ~ coef(lm(ri ~ rm, data = .x))[2])) %>% 
  unnest(betaest) %>% 
  ungroup() %>% 
  mutate(pfid = ntile(betaest, 20)) %>% 
  select(permno, pfid)

#creat match between permno and portfolio groups
#estimation period 1987-1991,testing period 1992-1995
df_withpfid <- df %>% 
  filter(between(year, 1987, 1995)) %>% 
  left_join(pf_groups)

#Portfolio Estimation

#estimate market betas and standard deviation of residuals
#Regressions start with the estimation period 1987-1991. 
#The beta and residual standard deviations are stored for the first year of testing period 
#which is the year 1992
beta_sigma_92 <- df_withpfid %>% 
  filter(year <= 1991) %>% 
  group_by(permno) %>% 
  nest() %>% 
  mutate(betas = map(data, ~ coef(lm(ri ~ rm, data = .x))[2]),
         sigmas = map(data, ~ sigma(lm(ri ~ rm, data = .x))),
         yeartesting = 1992) %>% 
  unnest(c(betas, sigmas)) %>% 
  select(permno, betas, sigmas, yeartesting)

#To get the estimates for the year 1992
#we can repeat the same procedure but including the additional year of 1992
testingperiod_92to95 <- data.frame()

for(i in 0:3){
  
  beta_sigma <- df_withpfid %>% 
    filter(year <= 1991 + i) %>% 
    group_by(permno) %>% 
    nest() %>% 
    mutate(betas = map(data, ~ coef(lm(ri ~ rm, data = .x))[2]),
           sigmas = map(data, ~ sigma(lm(ri ~ rm, data = .x))),
           yeartesting = 1992 + i) %>% 
    unnest(c(betas, sigmas)) %>% 
    select(permno, betas, sigmas, yeartesting)
  
  # estimates is for 1992
  testingperiod_92to95 <- bind_rows(testingperiod_92to95, beta_sigma)
  
}


testingperiod_92to95 <- rename(testingperiod_92to95, year = yeartesting)

# join with df_withpfid for all the testing years
df_testingperiod1 <- df_withpfid %>% 
  filter(between(year, 1992, 1995)) %>% 
  left_join(testingperiod_92to95) 

testing_period1 <- df_testingperiod1 %>% 
  group_by(pfid, year, month) %>% 
  summarise(rp = mean(ri, na.rm = T),
            betap = mean(betas, na.rm = T),
            betap_squared = mean(betas^2, na.rm = T),
            sigmap = mean(sigmas, na.rm = T)) %>% 
  ungroup() 

#Portfolio Testing

testing_period1_gammas <- testing_period1 %>% 
  group_by(year, month) %>% 
  nest(data = !c(year, month)) %>% 
  mutate(estim = map(data, ~lm(lead(rp) ~ betap + betap_squared + sigmap, data = .x)), 
         estim = map(estim, tidy)) %>% 
  unnest(estim) %>% 
  select(year, month, term, estimate) %>% 
  ungroup()

#Linearity
gamma_2t <- testing_period1_gammas %>% 
  filter(term == "betap_squared") %>% 
  pull(estimate)

# T test 
t.test(gamma_2t, mu = 0)

#No systematic effects of non-β risk
gamma_3t <- testing_period1_gammas %>% 
  filter(term == "sigmap") %>% 
  pull(estimate)

# T test 
t.test(gamma_3t)


#Positive expected return-risk tradeoff
gamma_1t <- testing_period1_gammas %>% 
  filter(term == "betap") %>% 
  pull(estimate)

# Rm
rm_data <- df_testingperiod1 %>% 
  filter(permno == 10145) %>% 
  pull(rm)

# Rm - R0
gamma_0t <- testing_period1_gammas %>% 
  filter(term == "(Intercept)") %>% 
  pull(estimate)

rm_r0 <- rm_data   - gamma_0t

t.test(gamma_1t, rm_r0) # t test 

length(gamma_1t)
length(rm_r0)

#Sharpe-Lintner (S-L) Hypothesis
rft <-  df_testingperiod1 %>% 
  filter(permno == 10145) %>% 
  pull(rf)

t.test(gamma_0t, rft)

Box.test(x = gamma_1t, lag = 1, type = "Ljung-Box")

Box.test(x = gamma_2t, lag = 1, type = "Ljung-Box")

Box.test(x = gamma_3t, lag = 1, type = "Ljung-Box")

# DATA VISUALIZATION 

# Risk Premium Over Time 

library(ggplot2)

   # Combine gamma estimates
gammas <- data.frame(
  year = rep(unique(testing_period1_gammas$year), 3),
  gamma = c(gamma_1t, gamma_2t, gamma_3t),
  type = rep(c("Gamma_1", "Gamma_2", "Gamma_3"), each = length(gamma_1t))
)

ggplot(gammas, aes(x = year, y = gamma, color = type)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = gamma - 1.96 * sd(gamma) / sqrt(length(gamma)),
                  ymax = gamma + 1.96 * sd(gamma) / sqrt(length(gamma)), fill = type), alpha = 0.2) +
  labs(title = "Risk Premium Over Time", x = "Year", y = "Gamma Estimates") +
  theme_minimal()

# Portfolio Beta Distribution 

ggplot(df_testingperiod1, aes(x = as.factor(pfid), y = betas)) +
  geom_boxplot() +
  labs(title = "Portfolio Beta Distribution", x = "Portfolio ID", y = "Beta") +
  theme_minimal()



