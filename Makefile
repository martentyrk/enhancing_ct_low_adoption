test:
	nose2 -v

hint:
	pytype dpfn

test-hint: test hint
	echo 'Finished running tests and checking type hints'

lint:
	pylint dpfn
	pycodestyle dpfn/*.py
	pycodestyle dpfn/**/*.py
	pydocstyle dpfn/*.py --ignore=D103,D104,D107,D203,D204,D213,D215,D400,D401,D404,D406,D407,D408,D409,D413
	pydocstyle dpfn/**/*.py --ignore=D103,D104,D107,D203,D204,D213,D215,D400,D401,D404,D406,D407,D408,D409,D413
