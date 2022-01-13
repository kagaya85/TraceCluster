# Skywalking trace data collect tool

build:
```shell
go build -o bin/collect
```

use:
```shell
# help
./bin/collect -h

# collect
./bin/collect -from "2022-01-01 00:00:00" -duration 12h -type success
```