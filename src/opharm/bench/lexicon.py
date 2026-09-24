POOLS = {
    "Host": {
        "dev": ["node-07", "node-12", "node-19", "node-23", "srv-04", "srv-15", "srv-26", "vm-31"],
        "heldout": ["node-42", "node-48", "node-55", "srv-37", "srv-61", "srv-68", "vm-09", "vm-14"],
    },
    "Database": {
        "dev": ["sql-03", "sql-07", "sql-11", "sql-18"],
        "heldout": ["sql-24", "sql-29", "sql-35", "sql-40"],
    },
    "Cluster": {
        "dev": ["k8s-eu1", "k8s-us2", "k8s-ap3", "k8s-eu4"],
        "heldout": ["k8s-sa1", "k8s-us7", "k8s-ca2", "k8s-eu9"],
    },
    "Account": {
        "dev": ["acme-ops", "acme-core", "acme-data", "acme-web"],
        "heldout": ["acme-infra", "acme-apps", "acme-edge", "acme-hub"],
    },
    "Repository": {
        "dev": ["gitops", "platform-config", "deploy-manifests", "helm-values"],
        "heldout": ["cluster-config", "release-manifests", "argo-apps", "flux-config"],
    },
}
REQUESTERS = {
    "dev": [("j.alvarez", "data platform"), ("p.okafor", "payments"), ("s.lindqvist", "growth"), ("r.mehta", "support")],
    "heldout": [("t.nguyen", "security"), ("a.kowalski", "billing"), ("f.haddad", "search"), ("y.tanaka", "mobile")],
}
ONCALL = {"dev": ["m.chen", "k.osei", "l.ferreira"], "heldout": ["d.novak", "h.sato", "e.brennan"]}
TICKET_BASE = {"dev": 4000, "heldout": 7000}
