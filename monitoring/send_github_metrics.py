
# Use authenticated requests in case of API rate limit exceeded.

import requests
import datetime
import os

HOST="51.107.91.115"
PORT="2003"

def get_metric(github_user, github_repo, metric_name):
    url = "https://api.github.com/repos/" + github_user + "/" + github_repo
    response = requests.get(url)
    ret = response.json()[metric_name]
    return ret

def get_issues(github_user, github_repo, is_closed, days_back):
    url = "https://api.github.com/repos/" + github_user + "/" + github_repo + "/issues?"
    if is_closed:
        url += "state=closed&"

    today = datetime.datetime.now()
    delta = datetime.timedelta(days = days_back)
    since_day = today - delta
    url += "since=" + str(since_day)

    response = requests.get(url)
    ret = response.json()[0]["url"]
    return ret

def send_issues(github_user, github_repo, days_back):
    data = get_issues(github_user, github_repo, True, days_back)
    stat = "github." + github_user + ".closed_issues"  + " " + str(data) + " `date +%s`"
    cmd = "echo " + stat + " | nc -q0 " + HOST + " " + PORT
    print(cmd)
    os.system(cmd)

    data = get_issues(github_user, github_repo, False, days_back)
    stat = "github." + github_user + ".opened_issues"  + " " + str(data) + " `date +%s`"
    cmd = "echo " + stat + " | nc -q0 " + HOST + " " + PORT
    print(cmd)
    os.system(cmd)

def send_metric(github_user, github_repo, metric_name):
    data = get_metric(github_user, github_repo, metric_name)
    stat = "github.eclipse." + metric_name + " " + str(data) + " `date +%s`"
    cmd = "echo " + stat + " | nc -q0 " + HOST + " " + PORT
    print(cmd)
    os.system(cmd)


categories = [
     ["eclipse","deeplearning4j", "stargazers_count", 7],
     ["eclipse","deeplearning4j", "forks_count", 7],
     ["KonduitAI","deeplearning4j", "stargazers_count", 7],
     ["KonduitAI","deeplearning4j", "forks_count", 30]
   ]

for cat in categories:
    send_metric(cat[0],cat[1], cat[2])
    send_issues(cat[0],cat[1], cat[3])
