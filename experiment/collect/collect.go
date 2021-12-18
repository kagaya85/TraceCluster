package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path"
	"strconv"
	"time"

	api "skywalking.apache.org/repo/goapi/query"
)

var (
	rootpath     string
	intervalStr  string
	startTimeStr string
	timezone     string
	url          string
)

const (
	swTimeLayout = "2006-01-02 150405"
)

type Config struct {
	startTime string
	endTime   string
	url       string
}

func init() {
	flag.StringVar(&startTimeStr, "startTime", "2021-12-17 00:00:00", "collect start time")
	flag.StringVar(&intervalStr, "interval", "24h", "collect interval duration")
	flag.StringVar(&timezone, "timezone", "Asia/Shanghai", "time zone")
	flag.StringVar(&rootpath, "root", "./raw", "root directory path where preprocessed data saved")
	flag.StringVar(&url, "url", "http://175.27.169.178:8080/graphql", "skywalking graphql server url")
	flag.Parse()
}

func main() {
	tz, err := time.LoadLocation(timezone)
	if err != nil {
		log.Fatalln("load time zone error:", err)
	}

	startTime, err := time.ParseInLocation("2006-01-02 15:04:05", startTimeStr, tz)
	if err != nil {
		log.Fatalln("parse time string error:", err)
	}

	interval, err := time.ParseDuration(intervalStr)
	if err != nil {
		log.Fatalln("parse interval duration error:", err)
	}

	cfg := &Config{
		startTime: startTime.Format(swTimeLayout),
		endTime:   startTime.Add(interval).Format(swTimeLayout),
		url:       url,
	}

	log.Printf("start collect span data from %s to %s", cfg.startTime, cfg.endTime)

	ctx := context.WithValue(context.Background(), urlKey{}, cfg.url)
	pageNum := 1
	needTotal := true

	condition := &api.TraceQueryCondition{
		QueryDuration: &api.Duration{
			Start: cfg.startTime,
			End:   cfg.endTime,
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
		log.Fatalf("query trace faild: %s", err)
	}
	traceIDs := make([]string, 0, traceBrief.Total)
	for _, trace := range traceBrief.Traces {
		traceIDs = append(traceIDs, trace.TraceIds...)
	}
	traces := QueryTraces(ctx, traceIDs)
	if err := SaveCSV(traces, "traces"); err != nil {
		log.Fatal(err)
	}
	log.Print("collect end")
}

func SaveCSV(traces []api.Trace, filename string) error {
	header := []string{"StartTime", "EndTime", "URL", "SpanType", "Service", "SpanId", "TraceId", "Peer", "ParentSpan", "Component", "IsError"}
	filepath := path.Join(rootpath, filename)
	var records [][]string

	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		records = make([][]string, 0, len(traces)+1)
		records = append(records, header)
	} else {
		records = make([][]string, 0, len(traces))
	}

	f, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}

	defer func() {
		if err := f.Close(); err != nil {
			log.Print(err)
		}
	}()

	for _, trace := range traces {
		for _, span := range trace.Spans {
			records = append(records, []string{
				strconv.Itoa(int(span.StartTime)),
				strconv.Itoa(int(span.EndTime)),
				*span.EndpointName,
				span.Type,
				span.ServiceCode,
				fmt.Sprintf("%s.%d", span.SegmentID, span.SpanID),
				span.TraceID,
				*span.Peer,
				fmt.Sprintf("%d", span.ParentSpanID), // TODO use segmentid.spanid
				*span.Component,
				strconv.FormatBool(*span.IsError),
			})
		}
	}

	w := csv.NewWriter(f)
	w.WriteAll(records)

	return w.Error()
}
