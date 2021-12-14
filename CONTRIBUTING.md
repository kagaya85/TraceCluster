## How to submit code using pull request

1. Checkout to a new branch in local
	```bash
	git checkout -b [branch name]
	```

1. Add and commit your change: 
	```bash
	git add . & git commit -m "some message"
	```

1. Push local new branch to remote repository and set to trach the remote
	```bash
	git push -u origin [branch name]
	```
	local branch name need to be same as the remote branch name

1. Then check this [link](https://github.com/kagaya85/TraceCluster/pulls) to make a new pull request

1. Choose the target branch (default is master) and your branch name, check the code changes

1. Click `Create pull request` and create a new PR

**Notice:** After create a PR, you can also commit new change to the new branch. After PR is merged, you can keep this branch and prepare for another PR.