float countIntersection(in rayQueryEXT rayQuery) {
    float numIntersections = 0.0;
    while(rayQueryProceedEXT(rayQuery)) {
        numIntersections += 1.0;
    }
    return numIntersections;
}