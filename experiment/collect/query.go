package main

import (
	"collect/assets"
	"context"
	"encoding/base64"
	"log"
	"time"

	"github.com/machinebox/graphql"
	"github.com/schollz/progressbar/v3"

	api "skywalking.apache.org/repo/goapi/query"
)

type (
	urlKey      struct{}
	usernameKey struct{}
	passwordKey struct{}
	authKey     struct{}
)

func QueryTrace(ctx context.Context, traceID string) (api.Trace, error) {
	var rsp map[string]api.Trace

	req := graphql.NewRequest(assets.Read("graphql/Trace.graphql"))
	req.Var("traceId", traceID)
	err := Execute(ctx, req, &rsp)

	return rsp["result"], err
}

func QueryTraces(ctx context.Context, traceIDs []string) <-chan api.Trace {
	traceC := make(chan api.Trace, 10)

	go func() {
		req := graphql.NewRequest(assets.Read("graphql/Trace.graphql"))
		setAuthorization(ctx, req)
		client := NewClient(ctx.Value(urlKey{}).(string))
		bar := progressbar.Default(int64(len(traceIDs)), "trace collecting")

		for _, traceID := range traceIDs {
			var rsp map[string]api.Trace

			req.Var("traceId", traceID)
			if err := client.Run(ctx, req, &rsp); err != nil {
				log.Printf("graphql execute error: %s", err)
				continue
			}

			traceC <- rsp["result"]
			bar.Add(1)
		}

		close(traceC)
	}()

	return traceC
}

func QueryTraceIDs(ctx context.Context, startTime time.Time, endTime time.Time) []string {
	interval := 15 * time.Minute
	itv := interval
	pageNum := 1
	needTotal := true

	traceIDs := []string{}

	for startTime.Before(endTime) && startTime.Before(time.Now()) {
		start := startTime.Format(swTimeLayout)
		end := startTime.Add(itv).Format(swTimeLayout)

		condition := &api.TraceQueryCondition{
			QueryDuration: &api.Duration{
				Start: start,
				End:   end,
				Step:  api.StepSecond,
			},
			TraceState: api.TraceStateAll,
			QueryOrder: api.QueryOrderByDuration,
			Paging: &api.Pagination{
				PageNum:   &pageNum,
				PageSize:  10000,
				NeedTotal: &needTotal,
			},
		}

		traceBrief, err := QueryBasicTraces(ctx, condition)
		if err != nil {
			itv -= 1 * time.Minute
			if itv <= 0 {
				log.Fatalf("query trace id faild: %s", err)
			}
			continue
		}

		if traceBrief.Total > 0 {
			log.Printf("get %d traceIDs from %q to %q", traceBrief.Total, start, end)
		}

		for _, trace := range traceBrief.Traces {
			traceIDs = append(traceIDs, trace.TraceIds...)
		}

		startTime = startTime.Add(itv)
		itv = interval
	}

	return traceIDs
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
