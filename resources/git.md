
## Installation
https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

## Get a private repository

```shell script
git clone https://username:password@github.com/denny5/SuperNova.git
```

## Setup git
#### Setup same user for all the repositories
```shell script
git config --global user.name "Name"
git config --global user.email "email@example.com"
```

#### Setup same user for only the current repositories
```shell script
git config user.name "Denny Wang"
git config user.email email@example.com
```

#### Check your settings
```shell script
git config --get-all user.name
git config --get user.email
```

## Basic usage
Read [Git Branching - Basic Branching and Merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)