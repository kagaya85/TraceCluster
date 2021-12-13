package collect

import (
	"collect/assets"
	"context"
	"log"

	"github.com/machinebox/graphql"

	api "skywalking.apache.org/repo/goapi/query"
)

type urlKey struct{}

func QueryTrace(ctx context.Context, traceID string) (api.Trace, error) {
	var rsp map[string]api.Trace

	req := graphql.NewRequest(assets.Read("graphql/Trace.graphql"))
	req.Var("traceId", traceID)
	err := Execute(ctx, req, rsp)

	return rsp["result"], err
}

func QueryTraces(ctx context.Context, traceIDs []string) []api.Trace {
	req := graphql.NewRequest(assets.Read("graphql/Trace.graphql"))
	client := NewClient(ctx.Value(urlKey{}).(string))

	traces := make([]api.Trace, 0, len(traceIDs))
	for traceID := range traceIDs {
		var rsp map[string]api.Trace

		req.Var("traceId", traceID)
		if err := client.Run(ctx, req, rsp); err != nil {
			log.Printf("graphql execute error: %s", err)
			continue
		}

		traces = append(traces, rsp["result"])
	}

	return traces
}

func QueryBasicTraces(ctx context.Context, condition *api.TraceQueryCondition) (api.TraceBrief, error) {
	var rsp map[string]api.TraceBrief

	req := graphql.NewRequest(assets.Read("graphql/Traces.graphql"))
	req.Var("condition", condition)
	err := Execute(ctx, req, &rsp)

	return rsp["result"], err
}

func NewClient(url string) *graphql.Client {
	client := graphql.NewClient(url)
	client.Log = func(message string) {
		log.Print(message)
	}

	return client
}

func Execute(ctx context.Context, req *graphql.Request, resp interface{}) error {
	client := NewClient(ctx.Value(urlKey{}).(string))

	return client.Run(ctx, req, resp)
}
