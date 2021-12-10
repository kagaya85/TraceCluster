package collect

type Config struct {
	startTime int
	endTime   int
	date      string
	url       string
}

func main() {
	// cfg := &Config{
	// 	date: "2021-12-10",
	// 	url:  "47.103.205.96:8080",
	// }
	// ctx := context.Background()
	// condition := &api.TraceQueryCondition{
	// 	ServiceID:         &serviceID,
	// 	ServiceInstanceID: &serviceInstanceID,
	// 	TraceID:           &traceID,
	// 	EndpointID:        &endpointID,
	// 	QueryDuration:     &duration,
	// 	MinTraceDuration:  nil,
	// 	MaxTraceDuration:  nil,
	// 	TraceState:        api.TraceStateAll,
	// 	QueryOrder:        order,
	// 	Tags:              tags,
	// 	Paging:            &paging,
	// }

	// traces, err := QueryTraces(ctx, condition)
	// if err != nil {
	// 	log.Fatalf("query trace faild: %s", err)
	// }

}
