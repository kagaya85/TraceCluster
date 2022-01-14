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
	rootpath    string
	durationStr string
	fromTimeStr string
	timezone    string
	url         string
	queryType   string
)

const (
	swTimeLayout = "2006-01-02 150405"
)

type Config struct {
	url      string
	username string
	password string
}

func init() {
	flag.StringVar(&fromTimeStr, "from", "2022-01-01 00:00:00", "collect from time")
	flag.StringVar(&durationStr, "duration", "24h", "collect duration time, support h/m/s")
	flag.StringVar(&timezone, "timezone", "Asia/Shanghai", "time zone")
	flag.StringVar(&rootpath, "savedir", "./raw", "directory path where preprocessed data saved")
	flag.StringVar(&url, "url", "http://localhost:8080/graphql", "skywalking graphql server url")
	flag.StringVar(&queryType, "type", "all", "collect traces type, support all/error/success")
	flag.Parse()
}

func main() {
	tz, err := time.LoadLocation(timezone)
	if err != nil {
		log.Fatalln("load time zone error:", err)
	}

	startTime, err := time.ParseInLocation("2006-01-02 15:04:05", fromTimeStr, tz)
	if err != nil {
		log.Fatalln("parse time string error:", err)
	}

	duration, err := time.ParseDuration(durationStr)
	if err != nil {
		log.Fatalln("parse interval duration error:", err)
	}
	endTime := startTime.Add(duration)

	cfg := &Config{
		url:      url,
		username: "basic-auth-username",
		password: "basic-auth-password",
	}

	log.Printf("start collect span data from %q to %q", startTime, endTime)

	ctx := context.Background()
	ctx = context.WithValue(ctx, urlKey{}, cfg.url)
	ctx = context.WithValue(ctx, usernameKey{}, cfg.username)
	ctx = context.WithValue(ctx, passwordKey{}, cfg.password)
	ctx = context.WithValue(ctx, authKey{}, "")

	var state api.TraceState
	switch queryType {
	case "all", "All", "ALL":
		state = api.TraceStateAll
	case "error", "Error", "ERROR":
		state = api.TraceStateError
	case "success", "Success", "SUCCESS":
		state = api.TraceStateSuccess
	default:
		log.Fatalf("invalid query trace type %q", queryType)
	}

	traceIDs := QueryTraceIDs(ctx, startTime, endTime, state)

	traceC := QueryTraces(ctx, traceIDs)

	start := startTime.Format("2006-01-02_15-04-05")
	if err := SaveCSV(traceC, fmt.Sprintf("%s_%s_%s", start, durationStr, "traces.csv")); err != nil {
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
