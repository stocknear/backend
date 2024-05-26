// Declare a route

function listToTree(comments, parentProp = "reply") {
    // Create id indexed comments dictionary
    const commentsDict = {};
    for (let comment of comments) {
        commentsDict[comment.id] = {
            ...comment,
            children: [],
        };
    }

    // Build the tree
    const tree = [];
    for (const comment of comments) {
        const parentId = comment[parentProp];
        if (parentId) {
            commentsDict[parentId].children.push(commentsDict[comment.id]);
        } else {
            tree.push(commentsDict[comment.id]);
        }
    }

    return tree;
}

module.exports = function (fastify, opts, done) {
    
    const pb = opts.pb;

    fastify.post('/get-all-comments', async (request, reply) => {
    const data = request.body;
    const postId = data?.postId
    let output;

    try {
        const result = await pb.collection("comments").getFullList({
            filter: `post="${postId}"`,
            expand: 'user,alreadyVoted(comment)',
            fields: "*,expand.user,expand.alreadyVoted(comment).user,expand.alreadyVoted(comment).type",
            sort: '-created',
        })

    
        output = listToTree(result);
    }
    catch(e) {
        output = [];
    }
    
+    reply.send({ items: output })
    });

    done();
};