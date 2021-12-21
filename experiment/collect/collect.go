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
	"strings"
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
	username  string
	password  string
}

func init() {
	flag.StringVar(&startTimeStr, "start-time", "2021-12-20 00:00:00", "collect start time")
	flag.StringVar(&intervalStr, "interval", "24h", "collect interval duration")
	flag.StringVar(&timezone, "timezone", "Asia/Shanghai", "time zone")
	flag.StringVar(&rootpath, "save-path", "./raw", "directory path where preprocessed data saved")
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
		username:  "basic-auth-username",
		password:  "basic-auth-password",
	}

	log.Printf("start collect span data from %q to %q", cfg.startTime, cfg.endTime)

	ctx := context.Background()
	ctx = context.WithValue(ctx, urlKey{}, cfg.url)
	ctx = context.WithValue(ctx, usernameKey{}, cfg.username)
	ctx = context.WithValue(ctx, passwordKey{}, cfg.password)
	ctx = context.WithValue(ctx, authKey{}, "")

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

	traceC := QueryTraces(ctx, traceIDs)

	timeNow := time.Now().Format("2006-01-02_15-04-05")
	if err := SaveCSV(traceC, fmt.Sprintf("%s_%s", timeNow, "traces.csv")); err != nil {
		log.Fatal(err)
	}
}

func SaveCSV(traceC <-chan api.Trace, filename string) error {
	header := []string{"StartTime", "EndTime", "URL", "SpanType", "Service", "SpanId", "TraceId", "Peer", "ParentSpan", "Component", "IsError"}
	bufsize := 1000
	records := make([][]string, 0, bufsize)

	if _, err := os.Stat(rootpath); os.IsNotExist(err) {
		os.MkdirAll(rootpath, os.ModePerm)
	}

	filepath := path.Join(rootpath, filename)
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		records = append(records, header)
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

	w := csv.NewWriter(f)
	for trace := range traceC {
		for _, span := range trace.Spans {
			var peer string
			if span.Type == "Exit" {
				peer = strings.Split(*span.Peer, ":")[0]
			} else {
				peer = span.ServiceCode
			}

			var parentSpanID string
			if span.ParentSpanID == -1 {
				if len(span.Refs) > 0 {
					s := span.Refs[0]
					parentSpanID = fmt.Sprintf("%s.%d", s.ParentSegmentID, s.ParentSpanID)
				} else {
					parentSpanID = "-1"
				}
			} else {
				parentSpanID = fmt.Sprintf("%s.%d", span.SegmentID, span.ParentSpanID)
			}

			// "StartTime", "EndTime", "URL", "SpanType", "Service", "SpanId", "TraceId", "Peer", "ParentSpan", "Component", "IsError
			records = append(records, []string{
				strconv.Itoa(int(span.StartTime)),
				strconv.Itoa(int(span.EndTime)),
				*span.EndpointName,
				span.Type,
				span.ServiceCode,
				fmt.Sprintf("%s.%d", span.SegmentID, span.SpanID),
				span.TraceID,
				peer,
				parentSpanID,
				*span.Component,
				strconv.FormatBool(*span.IsError),
			})

			if len(records) > bufsize {
				w.WriteAll(records)
				records = records[:0]
			}
		}
	}

	w.WriteAll(records)

	return w.Error()
}
