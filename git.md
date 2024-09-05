---
title: Git Reference
---

:::{card} Purpose
This reference provides instructions on how to use Git in order to submit homework assignments to Github Classroom for ECE 2390.
:::

## Step 1: Install Git

[Git](https://git-scm.com/)[^gitbook] is a DVCS popular in industry. In order to send and receive information from a remote server (e.g., [GitHub](https://github.com/)) you need Git installed on your local machine. Follow [these instructions](https://git-scm.com/downloads) to download Git for your operating system.

## Step 2: Create a GitHub account

A large percentage of Git repositories are hosted on [GitHub](https://github.com/). Follow [these instructions](https://github.com/signup) to set up a GitHub account.

## Step 3: Join the GitHub Classroom

When you accept an assignment from GitHub Classroom, the assignment template repository is forked and created for you. You'll need to be part of the classroom in order for this to work. Follow [the instructions on Canvas](https://canvas.pitt.edu/courses/272711) to join the GitHub Classroom.

## Step 4: Accept the assignment

Log onto Canvas and accept the homework assignment. There's an action that is performed when you accept the assignment which forks the homework template to your account.

## Step 5: Navigate to the area you want your local repo

`cd ece_2390`

## Step 6: Clone the fork

As an example:

`git clone https://github.com/SSOE-ECE1390/Homework0.git`

:::{note}
Each repository will have a unique url. You can find the unique url on GitHub.
:::

## Step 7: Do the homework

## Step 8: Add the changes to your local repo

`git add --all`

## Step 9: Commit the changes to your local repo

`git commit -m 'Some explaination of the changes'`

## Step 10: Push the changes to your forked repo

`git push origin main`

## Step 11: Verify your actions

`git status`

## Reference: Git CLI

:::{code} git
:caption: Example of Git CLI commands

C:\Users\jjcun\Documents\ece_2390>git clone https://github.com/SSOE-ECE1390/Homework0.git
C:\Users\jjcun\Documents\ece_2390>git add --all
C:\Users\jjcun\Documents\ece_2390>git commit -m 'Some explaination of the changes'
C:\Users\jjcun\Documents\ece_2390>git push origin main
C:\Users\jjcun\Documents\ece_2390>git status
:::

[^gitbook]: The [Pro Git book](https://git-scm.com/book/en/v2) written by Scott Chacon and Ben Straub is made entirely free and an incredibly useful reference. It is suggested that you read sections 1, 2, 3, and 6 to get comfortable with the program; however, at a minimum read sections 1 and 2.