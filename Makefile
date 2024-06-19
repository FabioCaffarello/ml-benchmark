WORKSPACE=ml-benchmark


guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

build-docs:
	npx nx graph --file=docs/dependency-graph/index.html
	npx nx  run-many --target=godoc --all

deploy-doc: build-docs
	poetry run mkdocs gh-deploy

lint:
	npx nx run-many --target=lint --all

check: guard-project cleanup
	npx nx test $(project)

check-all: cleanup
	npx nx run-many --target=test --all

run:
	docker-compose up -d

cleanup:
	@npx nx reset;
	@containers=$$(docker ps -q -a); \
	if [ -n "$$containers" ]; then \
		docker rm -f $$containers; \
	else \
		echo "No containers to remove"; \
	fi

PHONY: build-docs serve-doc deploy-doc lint check check-all cleanup