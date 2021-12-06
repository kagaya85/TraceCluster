package collect

import (
	"collect/assets"
	"context"
	"log"

	"github.com/machinebox/graphql"

	api "skywalking.apache.org/repo/goapi/query"
)

type urlKey struct{}

func QueryTraces(ctx context.Context, condition *api.TraceQueryCondition) (api.TraceBrief, error) {
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

func Execute(ctx context.Context, req *graphql.Request, rsp interface{}) error {
	client := NewClient(ctx.Value(urlKey{}).(string))

	return client.Run(ctx, req, rsp)
}
