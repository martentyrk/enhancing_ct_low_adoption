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
