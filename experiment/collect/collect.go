package collect

import (
	"context"
	"log"

	api "skywalking.apache.org/repo/goapi/query"
)

type Config struct {
	startTime int
	endTime   int
	data      string
	url       string
}

func main() {
	cfg := &Config{}
	ctx := context.Background()
	condition := &api.TraceQueryCondition{
		// ServiceID:         &serviceID,
		// ServiceInstanceID: &serviceInstanceID,
		// TraceID:           &traceID,
		// EndpointID:        &endpointID,
		// QueryDuration:     &duration,
		// MinTraceDuration:  nil,
		// MaxTraceDuration:  nil,
		// TraceState:        api.TraceStateAll,
		// QueryOrder:        order,
		// Tags:              tags,
		// Paging:            &paging,
	}

	traces, err := QueryTraces(ctx, condition)
	if err != nil {
		log.Fatalf("query trace faild: %s", err)
	}

}
