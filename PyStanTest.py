import pystan

# Stan code for a trivial model---a normal prior for
# theta, with no data or likelihood function.
model_code = '''
    parameters {real theta;}
    model {theta ~ normal(0,1);}
'''

# The following will invoke Stan to build and compile a C++
# library; it will take some time and report progress to
# the console.
model = pystan.StanModel(model_code=model_code)

# This will run an MCMC algorithm for 2000 steps, discarding
# the first half of the run as burn-in; it will report progress
# to the console.
results = model.sampling(n_jobs=1)

# Here we print a Monte Carlo estimate of the posterior mean
# for y; if all goes well it should be near 0.
thetas = results.extract()['theta']
print('Mean of posterior samples:', thetas.mean())
