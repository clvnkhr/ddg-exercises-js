"use strict";

/**
 * @module Projects
 */
class SimplicialComplexOperators {

        /** This class implements various operators (e.g. boundary, star, link) on a mesh.
         * @constructor module:Projects.SimplicialComplexOperators
         * @param {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:LinearAlgebra.SparseMatrix} A0 The vertex-edge adjacency matrix of <code>mesh</code>.
         * @property {module:LinearAlgebra.SparseMatrix} A1 The edge-face adjacency matrix of <code>mesh</code>.
         */
        constructor(mesh) {
                this.mesh = mesh;
                this.assignElementIndices(this.mesh);

                this.A0 = this.buildVertexEdgeAdjacencyMatrix(this.mesh);
                this.A1 = this.buildEdgeFaceAdjacencyMatrix(this.mesh);
        }

        /** Assigns indices to the input mesh's vertices, edges, and faces
         * @method module:Projects.SimplicialComplexOperators#assignElementIndices
         * @param {module:Core.Mesh} mesh The input mesh which we index.
         */
        assignElementIndices(mesh) {
                for (let col of [mesh.vertices, mesh.edges, mesh.faces]) {
                        let i = 0;
                        for (let elt of col) {
                                elt.i = i;
                                i++;
                        }
                }
        }

        /** Returns the vertex-edge adjacency matrix of the given mesh.
         * @method module:Projects.SimplicialComplexOperators#buildVertexEdgeAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The vertex-edge adjacency matrix of the given mesh.
         */
        buildVertexEdgeAdjacencyMatrix(mesh) {
                let T = new Triplet(mesh.edges.length, mesh.vertices.length);
                for (let e of mesh.edges) {
                        let v1 = e.halfedge.vertex;
                        let v2 = e.halfedge.twin.vertex;
                        T.addEntry(1, e.i, v1.i);
                        T.addEntry(1, e.i, v2.i);
                }
                return SparseMatrix.fromTriplet(T);

        }

        /** Returns the edge-face adjacency matrix.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeFaceAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The edge-face adjacency matrix of the given mesh.
         */
        buildEdgeFaceAdjacencyMatrix(mesh) {
                let T = new Triplet(mesh.faces.length, mesh.edges.length);
                for (let f of mesh.faces) {
                        for (let e of f.adjacentEdges()) {
                                T.addEntry(1, f.i, e.i);
                        }
                }
                return SparseMatrix.fromTriplet(T);
        }

        /** Returns a column vector representing the vertices of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildVertexVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |V| entries. The ith entry is 1 if
         *  vertex i is in the given subset and 0 otherwise
         */
        buildVertexVector(subset) {
                let V = DenseMatrix.zeros(this.mesh.vertices.length, 1);
                for (let v of subset.vertices) {
                        V.set(1, v, 0);
                }
                return V;
        }

        /** Returns a column vector representing the edges of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |E| entries. The ith entry is 1 if
         *  edge i is in the given subset and 0 otherwise
         */
        buildEdgeVector(subset) {
                let E = DenseMatrix.zeros(this.mesh.edges.length, 1);
                for (let e of subset.edges) {
                        E.set(1, e, 0)
                }
                return E
        }

        /** Returns a column vector representing the faces of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildFaceVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |F| entries. The ith entry is 1 if
         *  face i is in the given subset and 0 otherwise
         */
        buildFaceVector(subset) {
                let F = DenseMatrix.zeros(this.mesh.faces.length, 1);
                for (let f of subset.faces) {
                        F.set(1, f, 0)
                }
                return F
        }

        /** Returns the star of a subset.
         * @method module:Projects.SimplicialComplexOperators#star
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The star of the given subset.
         */
        star(subset) {
                // for each e,v want to add all f with e and all e',f' with v.
                let out = MeshSubset.deepCopy(subset);

                let newEs = this.A0.timesDense(this.buildVertexVector(out));
                let setE = new Set();
                for (let i = 0; i < this.mesh.edges.length; i++) {
                        if (newEs.get(i, 0) != 0) {
                                setE.add(i);
                        }
                }
                out.addEdges(setE);
                let newFs = this.A1.timesDense(this.buildEdgeVector(out));
                let setF = new Set();
                for (let i = 0; i < this.mesh.faces.length; i++) {
                        if (newFs.get(i, 0) != 0) {
                                setF.add(i);
                        }
                }
                out.addFaces(setF);

                return out;
        }

        /** Returns the closure of a subset.
         * @method module:Projects.SimplicialComplexOperators#closure
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The closure of the given subset.
         */
        closure(subset) {
                let out = MeshSubset.deepCopy(subset)
                // not OK configs: edges without corresponding vertices, faces without corresponding edges.
                // So, go through all the faces and add the edges, then go through all the edges and add vertices.
                let newEs = this.A1.transpose().timesDense(this.buildFaceVector(out));
                let setE = new Set();
                for (let i = 0; i < this.mesh.edges.length; i++) {
                        if (newEs.get(i, 0) != 0) {
                                setE.add(i);
                        }
                }
                out.addEdges(setE);
                let newVs = this.A0.transpose().timesDense(this.buildEdgeVector(out));
                let setV = new Set();
                for (let i = 0; i < this.mesh.vertices.length; i++) {
                        if (newVs.get(i, 0) != 0) {
                                setV.add(i);
                        }
                }
                out.addVertices(setV);
                return out;
        }

        /** Returns the link of a subset.
         * @method module:Projects.SimplicialComplexOperators#link
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The link of the given subset.
         */
        link(subset) {
                let cs = this.closure(this.star(subset));
                let sc = this.star(this.closure(subset));
                cs.deleteSubset(sc);
                return cs
        }

        /** Returns true if the given subset is a subcomplex and false otherwise.
         * @method module:Projects.SimplicialComplexOperators#isComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {boolean} True if the given subset is a subcomplex and false otherwise.
         */
        isComplex(subset) {
                return this.closure(subset).equals(subset);
        }


        set2arr(set) {
                let arr = [];
                for (let i of set) {
                        arr.push(i);
                }
                return arr
        }

        colVec2arr(colVec) {
                let arr = [];
                let k = colVec.nRows()
                for (let i = 0; i < k; i++) {
                        let cv = colVec.get(i, 0)
                        if (cv > 0) {
                                arr.push(i);
                        }
                }
                return arr
        }

        /** Returns the degree if the given subset is a pure subcomplex and -1 otherwise.
         * @method module:Projects.SimplicialComplexOperators#isPureComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {number} The degree of the given subset if it is a pure subcomplex and -1 otherwise.
         */
        isPureComplex(subset) {
                // either a 2-complex or a 1-complex or a 0 complex.
                // 2 complex => all edges border a face.
                // 1 complex => no faces and all vertices are connected by edges.
                // 0 complex => only isolated vertices.
                if (!subset.vertices | subset.vertices.size === 0) {
                        return -1 // I guess this is an edge case.
                }
                if (!subset.edges | subset.edges.size === 0) {
                        if (!subset.faces | subset.faces.size == 0) {
                                // only vertices exist: this is 0-complex.
                                return 0
                        }
                        return -1
                }
                // vertices and edges exist; check that all edges in subset join two vertices,
                // and also that there are no edges that connect missing vertices
                let connected_vs = this.A0.transpose().timesDense(this.buildEdgeVector(subset))
                // alert("Testing: V=["
                //         + this.set2arr(subset.vertices)
                //         + "], E=["
                //         + this.set2arr(subset.edges)
                //         + "], cvs=["
                //         + this.colVec2arr(connected_vs)
                //         + "]")
                for (let i = 0; i < this.mesh.vertices.length; i++) {
                        // if i is in subset.vertices, do a check

                        if (subset.vertices.has(i)) {
                                if (connected_vs.get(i, 0) === 0) {
                                        alert("Vertex " + i + " is not connected by edges.")
                                        return -1;
                                }
                        }
                        else {
                                // not in subset, so not connected by edges
                                if (connected_vs.get(i, 0) > 0) {
                                        alert("Vertex " + i + " should not be connected by edges.")
                                        return -1;
                                }
                        }
                }
                if (!subset.faces | subset.faces.size === 0) {
                        // found 1-complex
                        return 1
                }

                // alert("hi", "there")
                // faces exist, so check that all faces are connected to edges (and vertices, but this is already guaranteed by the above).
                let connected_es = this.A1.transpose().timesDense(this.buildFaceVector(subset))
                for (let i = 0; i < this.mesh.edges.length; i++) {
                        if (subset.edges.has(i)) {
                                if (connected_es.get(i, 0) === 0) {
                                        return -1;
                                }
                        }
                        else {
                                // not in subset, so not between faces
                                if (connected_es.get(i, 0) > 0) {
                                        return -1;
                                }
                        }
                }
                return 2
        }


        /** Returns the boundary of a subset.
         * @method module:Projects.SimplicialComplexOperators#boundary
         * @param {module:Core.MeshSubset} subset A subset of our mesh. We assume <code>subset</code> is a pure subcomplex.
         * @returns {module:Core.MeshSubset} The boundary of the given pure subcomplex.
         */
        boundary(subset) {
                // INFO: no need to check for pure complex property.
                // boundary is the closure of the set of simplices that are proper faces
                // of exactly one simplex of J.
                let order = this.isPureComplex(subset)
                let out = new MeshSubset()
                if (order <= 0) {
                        return out
                }
                if (order === 1) {
                        let vertCounts = this.A0.transpose().timesDense(this.buildEdgeVector(subset))
                        let boundary_vertices = new Set()
                        for (let i = 0; i < this.mesh.vertices.length; i++) {
                                if (vertCounts.get(i, 0) === 1) {
                                        boundary_vertices.add(i);
                                }
                        }
                        out.addVertices(boundary_vertices);
                        return out;
                }
                if (order === 2) {
                        let edgeCounts = this.A1.transpose().timesDense(this.buildFaceVector(subset))
                        let boundary_edges = new Set()
                        for (let i = 0; i < this.mesh.edges.length; i++) {
                                if (edgeCounts.get(i, 0) === 1) {
                                        boundary_edges.add(i);
                                }
                        }
                        out.addEdges(boundary_edges);
                        return this.closure(out);
                }


        }
}
