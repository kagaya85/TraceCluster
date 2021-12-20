package main

import (
	"collect/assets"
	"context"
	"encoding/base64"
	"log"

	"github.com/machinebox/graphql"
	"github.com/schollz/progressbar/v3"

	api "skywalking.apache.org/repo/goapi/query"
)

type urlKey struct{}
type usernameKey struct{}
type passwordKey struct{}
type authKey struct{}

func QueryTrace(ctx context.Context, traceID string) (api.Trace, error) {
	var rsp map[string]api.Trace

	req := graphql.NewRequest(assets.Read("graphql/Trace.graphql"))
	req.Var("traceId", traceID)
	err := Execute(ctx, req, &rsp)

	return rsp["result"], err
}

func QueryTraces(ctx context.Context, traceIDs []string) []api.Trace {
	req := graphql.NewRequest(assets.Read("graphql/Trace.graphql"))
	setAuthorization(ctx, req)
	client := NewClient(ctx.Value(urlKey{}).(string))

	traces := make([]api.Trace, 0, len(traceIDs))
	bar := progressbar.Default(int64(len(traceIDs)), "trace collecting")
	for _, traceID := range traceIDs {
		var rsp map[string]api.Trace

		req.Var("traceId", traceID)
		if err := client.Run(ctx, req, &rsp); err != nil {
			log.Printf("graphql execute error: %s", err)
			continue
		}

		traces = append(traces, rsp["result"])
		bar.Add(1)
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
	// client.Log = func(message string) {
	// 	log.Print(message)
	// }

	return client
}

func Execute(ctx context.Context, req *graphql.Request, resp interface{}) error {
	setAuthorization(ctx, req)
	client := NewClient(ctx.Value(urlKey{}).(string))
	return client.Run(ctx, req, resp)
}

func setAuthorization(ctx context.Context, req *graphql.Request) {
	username := ctx.Value(usernameKey{}).(string)
	password := ctx.Value(passwordKey{}).(string)
	authorization := ctx.Value(authKey{}).(string)

	if authorization == "" && username != "" && password != "" {
		authorization = "Basic " + base64.StdEncoding.EncodeToString([]byte(username+":"+password))
	}

	if authorization != "" {
		req.Header.Set("Authorization", authorization)
	}
}
