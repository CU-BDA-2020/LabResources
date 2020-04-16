# PyStan installation

[Stan](http://mc-stan.org/) (which takes its name in part from Stanislaw Ulam, a co-creator of the earliest Monte Carlo methods) is both a **probabilistic programming language** capable of concisely describing Bayesian graphical models, and a powerful **application** that can build C++ libraries (with bindings to other languages, including Python and R) that implement such models, including sophisticated optimization and posterior sampling capability. Since Stan builds C++ libraries and language bindings, it requires users to have mutually compatible compilers and interpreters, a potentially nontrivial software requirement that has complicated Stan usage in the past. Fortunately for Anaconda users, Anaconda developers have recently gone to considerable efforts to simplify the process by automatically providing platform-specific compilers to Anaconda users who install Stan.

We will use [PyStan: The Python Interface to Stan](https://pystan.readthedocs.io/en/latest/).

**Shedule your PyStan installation and testing appropriately.**  Depending on what software you may already have installed, some steps of this setup may require time-consuming downloads. And there may be installation problems with some platforms. The earlier you can discover problems, the more likely it is that we can find a fix in time for assignments that use Stan.

### Installing PyStan in the `bda20` environment

The following straightforward procedure has been tested on Ubuntu Linux 18 LTS, macOS 10.14 (Mojave), and Windows 10 (running in a VMware Fusion virtual machine). Besides PyStan, this procedure installs Anaconda-produced compilers (a different set for each platform, e.g., `clang` for macOS and `gcc` for Windows 10), along with custom compiler toolchain settings tailored to PyStan's needs.

**Important note:** To support PyStan, conda replaces system-provided compilation tools in your shell's search path, and changes default compiler settings. If you are not currently using a conda environment for your course work, I strongly recommend that you set up a `bda20` environment as described in the  Python instructions in the CourseInfo repo. By installing PyStan only in the `bda20` environment, you ensure that your system's default compilation tools and settings remain available in your shell when you have not activated the `bda20` environment.

**Note to macOS users:** My Macs all have Apple's Xcode and command-line tools installed. I don't know if these are necessary for PyStan; my suspicion is that they are *not* necessary. However, if you have trouble with the PyStan installation or test, try installing Apple's developer tools, as described at the end of this document.

**PyStan installation (all platforms):** Perform the following 5 steps in order:

1. Update the `conda` command-line package manager in a terminal/shell session by running:

```bash
$ conda update conda
```
2. **Important:** *Activate the `bda20` environment.*
3. Update the `anaconda` package in the current (`bda20`) environment:
```bash
$ conda update anaconda
```
4. Install PyStan in the current environment using `conda` (do *not* follow the PyStan documentation instructions, which recommend installation with `pip` rather than `conda`; the instructions appear to be out-of-date). Also, install the [ArviZ](https://arviz-devs.github.io/arviz/index.html) package for posterior sampler output analysis:
```bash
$ conda install pystan arviz
```
5. Test PyStan, as described below.



**Legacy note:** As explained above, an environment including PyStan uses Anaconda's compilation tools. If you are using the current version of `conda`, you may skip ahead to "Test PyStan." *If you are using an older version of `conda`*, on some platforms, `conda` will notify the user of this by displaying shell environment variables that are being set to support PyStan. In my testing, on Ubuntu 18 and Windows 10, conda hides these manipulations from the user, but on macOS, it displays the settings, (1) when PyStan is installed, (2) when an environment including PyStan is activated, and (3) when an environment is deactivated (in which case the settings are reset to default values).

For example, on installation and activation in macOS using older versions of `conda`, you may see a notification of environment variable settings resembling this:

```bash
INFO: deactivate_clang_osx-64.sh made the following environmental changes:
INFO: deactivate_clangxx_osx-64.sh made the following environmental changes:
+HOST=x86_64-apple-darwin13.4.0
INFO: activate_clang_osx-64.sh made the following environmental changes:
+AR=/Users/loredo/anaconda/envs/bda18b/bin/x86_64-apple-darwin13.4.0-ar
+AS=/Users/loredo/anaconda/envs/bda18b/bin/x86_64-apple-darwin13.4.0-as
+CC=x86_64-apple-darwin13.4.0-clang
+CFLAGS=-march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe
+CHECKSYMS=/Users/loredo/anaconda/envs/bda18b/bin/x86_64-apple-darwin13.4.0-checksyms
+CLANG=/Users/loredo/anaconda/envs/bda18b/bin/x86_64-apple-darwin13.4.0-clang
[quite a few similar lines follow...]
```

In a Linux shell, you may see a notification like this:

```bash
INFO: deactivate-gxx_linux-64.sh made the following environmental changes:
INFO: deactivate-binutils_linux-64.sh made the following environmental changes:
+HOST=x86_64-conda_cos6-linux-gnu
INFO: deactivate-gcc_linux-64.sh made the following environmental changes:
INFO: activate-gxx_linux-64.sh made the following environmental changes:
+CXXFLAGS=-fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe
+CXX=/home/loredo/anaconda2/envs/bda18/bin/x86_64-conda_cos6-linux-gnu-c++
[quite a few similar lines follow...]
```

This is expected behavior.



## Test PyStan

Test PyStan using a Python interpreter by running these commands (you can copy and paste them at a Python or IPython interpreter prompt, or download and run the accompanying [PyStanTest.py](PyStanTest.py) script, e.g., with `ipython -i PyStanTest.py`):

```python
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
```

In an IPython session, the output should look something like this (note that the compilation step will take up to a few minutes; PyStan is composing and compiling quite a bit of code, even for this simple example):
```
INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_6fa7f15b3d85e088d23232eb587e3d58 NOW.

Gradient evaluation took 5e-06 seconds
1000 transitions using 10 leapfrog steps per transition would take 0.05 seconds.
Adjust your expectations accordingly!


Iteration:    1 / 2000 [  0%]  (Warmup)
Iteration:  200 / 2000 [ 10%]  (Warmup)
Iteration:  400 / 2000 [ 20%]  (Warmup)
Iteration:  600 / 2000 [ 30%]  (Warmup)
Iteration:  800 / 2000 [ 40%]  (Warmup)
Iteration: 1000 / 2000 [ 50%]  (Warmup)
Iteration: 1001 / 2000 [ 50%]  (Sampling)
Iteration: 1200 / 2000 [ 60%]  (Sampling)
Iteration: 1400 / 2000 [ 70%]  (Sampling)
Iteration: 1600 / 2000 [ 80%]  (Sampling)
Iteration: 1800 / 2000 [ 90%]  (Sampling)
Iteration: 2000 / 2000 [100%]  (Sampling)

 Elapsed Time: 0.009932 seconds (Warm-up)
               0.008974 seconds (Sampling)
               0.018906 seconds (Total)

[There will be 4 sections like this, corresponding to 4 separate chains being run.]

Mean of posterior samples: 0.014537743894794548

[You won't get that precise value for the mean, but it should be near 0.]
```

The compilation step is time-consuming, but for this trivial model, running the sampler is very fast. If you'd like to run the sampler again (maybe to double-check that the mean is near zero), just keep the IPython session open, and enter the `results = ...` , `thetas = ...`, and `print...` lines again.

## macOS developer tools

Mac users who do not have Apple's developer tools installed and who have problems with PyStan should try installing Apple's command-line tools, and XQuartz.

### Xcode and command-line tools for macOS

The standard way to get Apple's command-line tools is to install Xcode, Apple's developer enviroment. That said, Xcode is large, and if you don't plan on developing macOS or iOS apps, you may want to install the command-line tools alone. However, to do that, you must have an Apple developer account (it's free). If you want to install the tools alone, you'll find the installer at [Apple Developer downloads](https://developer.apple.com/download/more/) (login required). Otherwise, install the tools by installing Xcode, as follows:

- Download Xcode using the Mac App Store app.  The current version is Xcode 11.3.  My current PyStan scripts have run on systems with Xcode 9, so you probably need not upgrade Xcode if you have a slightly older version already installed.  *Note that Xcode is large and the download can be time consuming (sometimes taking hours)—don't postpone this until the last minute.*
- *Launch Xcode.*  You must launch it after installing; it installs the command-line tools after its first launch. You may then quit it.

### XQuartz

XQuartz is a macOS implementation of a large suite of Linux tools, particularly tools supporting Linux-style interaction with GUI windows in the macOS environment; it enables many cross-platform Linux tools to run in macOS. *I don't believe it is a requirement for our environment*, but `matplotlib` does interact with it (for fonts) if it is present. If you are an XQuartz user, make sure you are running the latest version. You can install the newest XQuartz from: [XQuartz](https://www.xquartz.org/).

